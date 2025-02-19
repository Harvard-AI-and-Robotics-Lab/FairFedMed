import os.path as osp

import copy, math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.engine.trainer import TrainerX, create_ddp_model
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from evaluation.metrics import compute_auc

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'GLP_OT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GLP_OT.N_CTX
        ctx_init = cfg.TRAINER.GLP_OT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.GLP_OT.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.GLP_OT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype) 
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1) 
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION


    def forward(self):
       
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        
        ctx = ctx.permute(1, 0, 2, 3) 
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=0.04):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        # Create low-rank adaptation matrices
        # self.lora_A = nn.Parameter(original_linear.weight.new_zeros((original_linear.in_features, rank)))
        # self.lora_B = nn.Parameter(original_linear.weight.new_zeros((rank, original_linear.out_features)))

        self.lora_A = nn.Embedding(original_linear.in_features, rank)
        self.lora_B = nn.Embedding(rank, original_linear.out_features)

        device = self.original_linear.weight.device
        dtype = self.original_linear.weight.dtype
        # Ensure lora_B is on the same device and dtype as linear weights
        self.lora_A.weight.data = self.lora_A.weight.data.to(dtype).to(device)
        self.lora_B.weight.data = self.lora_B.weight.data.to(dtype).to(device)

        # Set self.linear parameters to not require gradients
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
        nn.init.zeros_(self.lora_A.weight)  
        nn.init.normal_(self.lora_B.weight)  
    
    def weight(self, x, attr=None):
        return self.original_linear.weight + self.scaling*(self.lora_A.weight @ self.lora_B.weight).t()

    def bias(self):
        return self.original_linear.bias

    def forward(self, x, attr=None):
        return self.original_linear(x) + ((x @ self.lora_A.weight) @ self.lora_B.weight) * self.scaling

    def save_lora_weights(self):
        return {
            'lora_A': self.lora_A.weight.data.clone(),
            'lora_B': self.lora_B.weight.data.clone()
        }

    def load_lora_weights(self, lora_weights):
        self.lora_A.data.copy_(lora_weights['lora_A'])
        self.lora_B.data.copy_(lora_weights['lora_B'])


class SVLoRALinear(nn.Module):
    def __init__(
        self, 
        original_linear, 
        rank=4, 
        alpha=0.4,
        global_s=False,
    ):
        super(SVLoRALinear, self).__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        self.global_s = global_s

        self.lora_A = nn.Embedding(original_linear.in_features, rank)
        self.lora_S = nn.Embedding(rank, 1)
        if self.global_s:
            self.lora_S_global = nn.Embedding(rank, 1)
        self.lora_B = nn.Embedding(rank, original_linear.out_features)

        device = self.original_linear.weight.device
        dtype = self.original_linear.weight.dtype
        # Ensure lora_B is on the same device and dtype as linear weights
        self.lora_A.weight.data = self.lora_A.weight.data.to(dtype).to(device)
        self.lora_S.weight.data = self.lora_S.weight.data.to(dtype).to(device)
        if self.global_s:
            self.lora_S_global.weight.data = self.lora_S_global.weight.data.to(dtype).to(device)
        self.lora_B.weight.data = self.lora_B.weight.data.to(dtype).to(device)

        # Set self.linear parameters to not require gradients
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
        nn.init.zeros_(self.lora_A.weight)  
        # Set lora_S weights to a linear space from 1.0 to 0.1
        lora_s_steps = len(self.lora_S.weight)
        self.lora_S.weight.data = torch.linspace(
            1, 0.1, steps=lora_s_steps, 
            device=self.lora_S.weight.device,
        ).to(self.lora_S.weight.dtype)
        if self.global_s:
            self.lora_S_global.weight.data = torch.linspace(
                1, 0.1, steps=lora_s_steps, 
                device=self.lora_S_global.weight.device,
            ).to(self.lora_S_global.weight.dtype)
        nn.init.normal_(self.lora_B.weight)  
    
    def forward(self, x, attr=None):
        if self.global_s:
            return self.original_linear(x) + (((x @ self.lora_A.weight) @ torch.diag(self.lora_S.weight + self.lora_S_global.weight)) @ self.lora_B.weight) * self.scaling
        else:
            return self.original_linear(x) + (((x @ self.lora_A.weight) @ torch.diag(self.lora_S.weight)) @ self.lora_B.weight) * self.scaling
        
    def save_lora_weights(self):
        w = {
            'lora_A': self.lora_A.data.clone(),
            'lora_S': self.lora_S.data.clone(),
            'lora_B': self.lora_B.data.clone()
        }
        if self.global_s:
            w['lora_s_global'] = self.lora_S_global.data.clone()

        return w

    def load_lora_weights(self, lora_weights):
        self.lora_A.data.copy_(lora_weights['lora_A'])
        self.lora_S.data.copy_(lora_weights['lora_S'])
        self.lora_B.data.copy_(lora_weights['lora_B'])
        if self.global_s:
            self.lora_S_global.data.copy_(lora_weights['lora_S_global'])


class FairLoRALinear(nn.Module):
    def __init__(
        self, 
        original_linear, 
        rank=4, 
        alpha=0.4,
        global_s=False,
        num_attrs=-1,  # num_groups
    ):
        super(FairLoRALinear, self).__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        self.global_s = global_s
        assert num_attrs > 0, 'Number of attributes must be provided!'
        self.num_attrs = num_attrs

        if original_linear.weight.dim() == 2:
            self.is_1x1_conv = False
            in_features = original_linear.in_features
            out_features = original_linear.out_features
        else:
            self.is_1x1_conv = True
            out_features, in_features = original_linear.weight.shape[:2]
            
        self.lora_A = nn.Embedding(in_features, rank)
        self.lora_S = nn.Embedding(num_attrs, rank)
        if self.global_s:
            self.lora_S_global = nn.Embedding(1, rank)
        self.lora_B = nn.Embedding(rank, out_features)

        device = self.original_linear.weight.device
        dtype = self.original_linear.weight.dtype
        # Ensure lora_B is on the same device and dtype as linear weights
        self.lora_A.weight.data = self.lora_A.weight.data.to(dtype).to(device)
        self.lora_S.weight.data = self.lora_S.weight.data.to(dtype).to(device)
        if self.global_s:
            self.lora_S_global.weight.data = self.lora_S_global.weight.data.to(dtype).to(device)
        self.lora_B.weight.data = self.lora_B.weight.data.to(dtype).to(device)

        # Set self.linear parameters to not require gradients
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self, init_type='same+cycle'):
        # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
        nn.init.zeros_(self.lora_A.weight)  
        # Set lora_S weights to a linear space from 1.0 to 0.1
        rank = self.lora_S.weight.shape[-1]
        if init_type in {'same', 'cycle_shift'}:
            lora_S_weight = torch.linspace(
                1, 0.1, steps=rank, 
                device=self.lora_S.weight.device,
            ).to(self.lora_S.weight.dtype)
            if init_type == 'same':
                self.lora_S.weight.data = lora_S_weight[None].repeat(self.num_attrs,1)
            else:
                assert rank >= self.num_attrs
                self.lora_S.weight.data = torch.stack([
                    torch.cat([
                        lora_S_weight[i*(rank//self.num_attrs):], 
                        lora_S_weight[:i*(rank//self.num_attrs)]
                    ])
                    for i in range(self.num_attrs)
                ])
        else:
            assert rank % 2 == 0 and rank >= self.num_attrs
            lora_S_weight = torch.linspace(
                0.5, 0.1, steps=rank//2, 
                device=self.lora_S.weight.device,
            ).to(self.lora_S.weight.dtype)
            cycle_weight = torch.stack([
                torch.cat([
                    lora_S_weight[i*(int(0.5*rank)//self.num_attrs):], 
                    lora_S_weight[:i*(int(0.5*rank)//self.num_attrs)]
                ])
                for i in range(self.num_attrs)
            ])
            self.lora_S.weight.data = torch.cat([
                (lora_S_weight[None]).repeat(self.num_attrs,1), cycle_weight*0.2
            ], dim=1)
        if self.global_s:
            self.lora_S_global.weight.data = torch.linspace(
                1, 0.1, steps=rank, 
                device=self.lora_S_global.weight.device,
            ).to(self.lora_S_global.weight.dtype)
        nn.init.normal_(self.lora_B.weight)  
    
    def weight(self, x, attr):
        with torch.no_grad():
            attr_one_hot = F.one_hot(
                attr, num_classes=self.num_attrs
            ).to(x.device).to(x.dtype)  # bs x num_attrs
        lora_S = attr_one_hot @ self.lora_S.weight            # bs x r
        lora_S = torch.stack([torch.diag(s) for s in lora_S]) # bs x r x r
        if self.global_s:
            lora_S = lora_S + torch.diag(self.lora_S_global.weight)
        # oct b-scan data will be splited into multiple slices
        num_slices = x.shape[1] // lora_S.shape[0]
        lora_S = lora_S[:,None].repeat(1,num_slices,1,1).flatten(0,1)

        # b x c_in x c_out
        dw = torch.einsum('cr, brr->bcr', self.lora_A.weight, lora_S) @ self.lora_B.weight
        # b x c_out x c_in
        dw = self.scaling * dw.permute(0,2,1)
        return self.original_linear.weight[None].repeat(dw.shape[0],1,1) + dw
    
    def bias(self):
        return self.original_linear.bias
        
    def forward(self, x, attr):
        y = self.original_linear(x)

        with torch.no_grad():
            attr_one_hot = F.one_hot(
                attr, num_classes=self.num_attrs
            ).to(x.device).to(x.dtype)  # bs x num_groups
            
            lambda_group = 0.7
            attr_one_hot = attr_one_hot * lambda_group + (1 - attr_one_hot) * (1-lambda_group)/(self.num_attrs-1)

        lora_S = attr_one_hot @ self.lora_S.weight            # bs x r
        lora_S = torch.stack([torch.diag(s) for s in lora_S]) # bs x r x r
        if self.global_s:
            lora_S = lora_S + torch.diag(self.lora_S_global.weight)
        if self.is_1x1_conv:
            b, c_in, h, w = x.shape
            x = x.reshape(b, c_in, h*w).permute(2,0,1)

        # oct b-scan data will be splited into multiple slices
        num_slices = x.shape[1] // lora_S.shape[0]
        lora_S = lora_S[:,None].repeat(1,num_slices,1,1).flatten(0,1)

        dy = torch.einsum('nbr,brr->nbr', x @ self.lora_A.weight, lora_S)
        dy = (dy @ self.lora_B.weight) * self.scaling
        if self.is_1x1_conv:
            dy = dy.reshape(h, w, b, -1).permute(2,3,0,1)
  
        return y + dy
        
    def save_lora_weights(self):
        w = {
            'lora_A': self.lora_A.data.clone(),
            'lora_S': self.lora_S.data.clone(),
            'lora_B': self.lora_B.data.clone()
        }
        if self.global_s:
            w['lora_s_global'] = self.lora_S_global.data.clone()

        return w

    def load_lora_weights(self, lora_weights):
        self.lora_A.data.copy_(lora_weights['lora_A'])
        self.lora_S.data.copy_(lora_weights['lora_S'])
        self.lora_B.data.copy_(lora_weights['lora_B'])
        if self.global_s:
            self.lora_S_global.data.copy_(lora_weights['lora_S_global'])

# Function to apply LoRA to linear layers
def apply_lora_to_model(
    model,
    unfreeze_image_encoder,
    rank=4, 
    alpha=0.04, 
    lora_type='loRA', 
    global_s=False, 
    num_attrs=-1,
):
    named_modules = {name: module for name, module in model.named_modules()}
    for name, module in named_modules.items():
        if unfreeze_image_encoder and name.startswith('image_encoder.'):
            # vit backbone
            if isinstance(module, nn.Linear) and '.mlp.' in name:
                idx = name.split('.').index('resblocks') + 1
                layer = int(name.split('.')[idx])

                # Replace the original linear layer with LoRA adapted layer
                if lora_type == 'LoRA':
                    lora_layer = LoRALinear(
                        module, rank=rank, alpha=alpha
                    )
                elif lora_type == 'SVLoRA':
                    lora_layer = SVLoRALinear(
                        module, rank=rank, alpha=alpha, global_s=global_s
                    )
                elif lora_type == 'FairLoRA':
                    lora_layer = FairLoRALinear(
                        module, rank=rank, alpha=alpha, global_s=global_s, num_attrs=num_attrs
                    )
                else:
                    raise NotImplementedError
                # Replace the module in the model
                parent_module = model
                # Navigate to the parent module
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name.split('.')[-1], lora_layer)
            
            elif name.startswith('image_encoder.layer') or name.startswith('image_encoder.attnpool'):
                # resnet backbone
                # image_encoder.layer1.0.conv1.weight torch.Size([64, 64, 1, 1])
                # image_encoder.layer1.0.bn1.weight torch.Size([64])
                # image_encoder.layer1.0.bn1.bias torch.Size([64])
                # image_encoder.layer1.0.conv2.weight torch.Size([64, 64, 3, 3])
                # image_encoder.layer1.0.bn2.weight torch.Size([64])
                # image_encoder.layer1.0.bn2.bias torch.Size([64])
                # image_encoder.layer1.0.conv3.weight torch.Size([256, 64, 1, 1])
                # image_encoder.layer1.0.bn3.weight torch.Size([256])
                # image_encoder.layer1.0.bn3.bias torch.Size([256])
                # image_encoder.layer1.0.downsample.0.weight torch.Size([256, 64, 1, 1])
                # image_encoder.layer1.0.downsample.1.weight torch.Size([256])
                # image_encoder.layer1.0.downsample.1.bias torch.Size([256])
                if (isinstance(module, nn.Conv2d) and 'conv' in name and module.weight.shape[-2:] == (1, 1)) \
                    or ('attnpool' in name and isinstance(module, nn.Linear)):
                    if 'attnpool' in name:
                        lora_layer = LoRALinear(
                            module, rank=rank, alpha=alpha
                        )
                    elif lora_type == 'FairLoRA':
                        lora_layer = FairLoRALinear(
                            module, rank=rank, alpha=alpha, global_s=global_s, num_attrs=num_attrs
                        )
                    else:
                        raise NotImplementedError
                    # Replace the module in the model
                    parent_module = model
                    # Navigate to the parent module
                    for part in name.split('.')[:-1]:
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, name.split('.')[-1], lora_layer)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.pixel_mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN)
        self.pixel_std = torch.tensor(self.cfg.INPUT.PIXEL_STD)

        self.n_cls = len(classnames)
        # Check if the dataset modality involves 3D input
        self.is_3d_input = cfg.DATASET.MODALITY_TYPE in {'oct_bscans', 'oct_bscans_3d'}
        if self.is_3d_input:
            self.dim_per_3d_slice = cfg.DATASET.DIM_PER_3D_SLICE 
            self.proj_per_3d_slice = nn.Conv2d(in_channels=self.dim_per_3d_slice, 
                                       out_channels=3, 
                                       kernel_size=5, 
                                       padding=2,
                                       dtype=clip_model.dtype)
            # Initialize the weights and biases
            std = self.dim_per_3d_slice ** -0.5
            nn.init.normal_(self.proj_per_3d_slice.weight, std=std)
            nn.init.zeros_(self.proj_per_3d_slice.bias)

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:0")
        self.device1 = torch.device("cuda")
        self.N = cfg.TRAINER.GLP_OT.N
        self.dataset = cfg.DATASET.NAME
        self.use_uniform = True
        self.eps = cfg.TRAINER.GLP_OT.EPS
        self.max_iter = 100
        self.thresh = cfg.TRAINER.GLP_OT.THRESH
        self.OT = cfg.TRAINER.GLP_OT.OT
        self.top_percent = cfg.TRAINER.GLP_OT.TOP_PERCENT
        self.max_iter = cfg.TRAINER.GLP_OT.MAX_ITER

    def Sinkhorn(self, K, u, v):
        '''
        K is the Wasserstein distance, [bs*n_cls, 196, 77]
        u is , [bs*n_cls, 196]
        v is , [bs*n_cls, 77]
        '''
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version
        a is the source prob, [bs*n_cls, 196]
        b is the target prob, [bs*n_cls, 77]
        M is the cost matrix, i.e. Wasserstein distance, [bs*n_cls, 196, 77]

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {'err': []}
        stopThr=self.thresh 

        # K = torch.exp(M / (-reg))
        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while (cpt < numItermax):

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() <  stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u, dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v, dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

    def forward(self, image, attr=None):
        b, c, h, w = image.shape
        if self.cfg.DATASET.NAME == "FairFedMed":
            image = image / 255.
            if self.is_3d_input:
                # split 3d input into multiple slices to process
                image = image.reshape(-1, self.dim_per_3d_slice, h, w)
                image = self.proj_per_3d_slice(image.type(self.dtype))

                # # Find the minimum and maximum values per batch
                min_vals = image.amin(dim=(1, 2, 3), keepdim=True)
                max_vals = image.amax(dim=(1, 2, 3), keepdim=True)
                # Normalize to range [0, 1]
                image = (image - min_vals) / (max_vals - min_vals + 1e-5)  

            image = image - self.pixel_mean.reshape(1,-1,1,1).to(image.device)
            image = image / self.pixel_std.reshape(1,-1,1,1).to(image.device)

        image_features = self.image_encoder(image.type(self.dtype), attr=attr)  
        image_feature_pool = image_features[0]
        image_features = image_features[1:]  
        M = image_features.shape[0]  # 14 * 14
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()   
        tokenized_prompts = self.tokenized_prompts
        if self.dataset == "ImageNet":
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1)) 
            text_features = text_features.to(self.device)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts) 
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        
        image_features =  F.normalize(image_features, dim=2) 
        image_feature_pool = F.normalize(image_feature_pool, dim=1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M, self.N, -1)  # num_pixels, 2,  batch_size * n_cls
        sim = sim.permute(2,0,1)       # batch_size * n_cls, num_pixels, 2
        wdist = 1.0 - sim

        xx = torch.zeros(sim.shape[0], M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        if self.OT == 'Sinkhorn':
            yy = torch.zeros(sim.shape[0], self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        elif self.OT == 'COT':
            top_percent = min(torch.sum(xx).item(), self.top_percent)
            yy = torch.zeros(sim.shape[0], self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N) * top_percent
        elif self.OT == 'None':
            pass
        else:
            raise NotImplementedError

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == 'Sinkhorn':
                T = self.Sinkhorn(KK, xx, yy)  # T is the transport plan
                if torch.isnan(T).any():
                    return None
            elif self.OT == 'COT':
                T = self.entropic_COT_fast(xx, yy, KK,0.01,numItermax=self.max_iter)
                if torch.isnan(T).any():
                    return None
            elif self.OT == 'None':
                T = 1
            else:
                raise NotImplementedError

        if self.OT == 'None':
            sim_op = torch.mean(T * sim, dim=(1, 2))
        else:
            sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, -1, self.n_cls)
        sim_op = sim_op.mean(1)  # average all slices 
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op   
        
        return logits


# @TRAINER_REGISTRY.register()
class GLP_OT_SVLoRA(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GLP_OT.PREC in ["fp16", "fp32", "amp"]
    
    def retrieval_attributes(self, attr_name):
        return {
            'race': ['Asian', 'Black', 'White'],
            'language': ['English', 'Spanish', 'Others'],
            'ethnicity': ['Non-hispanic', 'Hispanic'],
        }[attr_name]

    def _get_layer_by_name(self, param_name):
        """
        Utility function to retrieve the layer/module by its parameter name.
        This function is used to identify the type of layer (e.g., BatchNorm2d) for conditional gradient updates.
        """
        # Split the parameter name by dots and navigate through the model structure
        modules = param_name.split(".")
        module = self.model
        for mod in modules[:-1]:  # Navigate through all submodules except the final parameter
            module = getattr(module, mod)
        return module

    def build_model(self):
        cfg = self.cfg
        self.pixel_mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN)
        self.pixel_std = torch.tensor(self.cfg.INPUT.PIXEL_STD)

        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GLP_OT.PREC == "fp32" or cfg.TRAINER.GLP_OT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()   

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "proj_per_3d_slice" in name:
                param.requires_grad_(True)
            elif isinstance(self._get_layer_by_name(name), nn.BatchNorm2d): 
                # and ('bn1' in name or 'bn3' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        apply_lora_to_model(
            model=self.model,
            unfreeze_image_encoder=self.cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER,
            rank=self.cfg.TRAINER.GLP_OT_LORA.RANK,
            alpha=self.cfg.TRAINER.GLP_OT_LORA.ALPHA,
            lora_type=self.cfg.TRAINER.GLP_OT_LORA.TYPE,
            global_s=self.cfg.TRAINER.GLP_OT_LORA.GLOBAL_S,
            num_attrs=len(self.retrieval_attributes(self.cfg.DATASET.ATTRIBUTE_TYPE))
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, 'grad:', param.requires_grad, param.shape)
            if "prompt_learner" in name or "proj_per_3d_slice" in name:
                print(name, 'grad:', param.requires_grad, param.shape)

        if cfg.DATASET.NAME== "ImageNet":
            self.device =  torch.device("cuda:0")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)
        
        params_to_optimize = list(self.model.prompt_learner.parameters()) + \
                list(self.model.image_encoder.parameters())
        if self.model.is_3d_input:
            params_to_optimize += list(self.model.proj_per_3d_slice.parameters())
        self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # Register the prompt learner
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        if cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER:
            # Register the image encoder
            self.register_model("image_encoder", self.model.image_encoder, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GLP_OT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        self.model = create_ddp_model(self.model, broadcast_buffers=False)

    def forward_backward(self, batch, is_last_client=False):
        if self.cfg.DATASET.NAME == "FairFedMed":
            image, label, _, attr = self.parse_batch_train(batch)
        else:
            image, label = self.parse_batch_train(batch)
            attr = None

        prec = self.cfg.TRAINER.GLP_OT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, attr)
            # loss = F.cross_entropy(output, label)
            if attr is None:
                loss = F.cross_entropy(output, label)
            else:
                cls_loss = F.cross_entropy(output, label)

                fairness_loss_type = "confidence"  # confidence

                # Initialize accuracy dictionary
                unique_attr_labels = torch.unique(attr)  # Get all the unique demographic group labels
                
                if fairness_loss_type == "acc":
                    group_accuracy = {}
                    # Calculate accuracy for each group
                    for attr_label in unique_attr_labels:
                        group_mask = (attr == attr_label)
                        group_predictions = output[group_mask].argmax(dim=1)
                        group_labels = label[group_mask]
                        group_accuracy[attr_label.item()] = (group_predictions == group_labels).float().mean()
                    
                    group_accuracy = torch.tensor(list(group_accuracy.values()))

                    # Fairness loss: penalize deviations of group accuracy from the average accuracy
                    fairness_loss = torch.mean(torch.abs(group_accuracy - group_accuracy.mean()))
                
                else:
                    group_confidence = {}

                    # Convert logits to soft labels (probabilities)
                    probs = F.softmax(output, dim=1)  # Shape (N, C)
                    
                    # Get predicted probability for the correct class (soft label)
                    correct_probs = probs[torch.arange(len(label)), label]  # Shape (N,)

                    for group in unique_attr_labels:
                        group_mask = (attr == group)
                        group_confidence[group.item()] = 1 - correct_probs[group_mask].mean()  # Average confidence per group
                    
                    # Compute fairness regularization: minimize confidence gap across groups
                    group_confidence = torch.tensor(list(group_confidence.values()))
                    fairness_loss = torch.mean(torch.abs(group_confidence - group_confidence.mean()))
        
                # Weight for fairness regularization
                lambda_fairness = self.cfg.TRAINER.LAMBDA_FAIRNESS  # You can adjust this to control the fairness strength
                loss = cls_loss + lambda_fairness * fairness_loss
                
            self.model_backward_and_update(loss)

        if output.shape == label.shape:
            output_prob = output.sigmoid()
        else:
            output_prob = output.softmax(-1)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if self.cfg.DATASET.NAME == "FairFedMed":
            if len(set(label)) == 1:
                loss_summary["auc"] = 1
            else:
                loss_summary["auc"] = compute_auc(output_prob, label).item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        if self.cfg.DATASET.NAME == "FairFedMed":
            # input = input / 255.
            # input = input - self.pixel_mean.reshape(1,-1,1,1).to(input.device)
            # input = input / self.pixel_std.reshape(1,-1,1,1).to(input.device)

            attrs = batch["attrs"].t()
            tgt_attr_idx = self.cfg.DATASET.ATTRIBUTES.index(self.cfg.DATASET.ATTRIBUTE_TYPE) 
        
            return input, label, attrs, attrs[tgt_attr_idx]
        else:
            return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)

        if self.cfg.DATASET.NAME == "FairFedMed":
            # input = input / 255.
            # input = input - self.pixel_mean.reshape(1,-1,1,1).to(input.device)
            # input = input / self.pixel_std.reshape(1,-1,1,1).to(input.device)

            attrs = batch["attrs"].t()
            tgt_attr_idx = self.cfg.DATASET.ATTRIBUTES.index(self.cfg.DATASET.ATTRIBUTE_TYPE) 
            
            return input, label, attrs, attrs[tgt_attr_idx]
        else:
            return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)