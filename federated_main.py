import os
import argparse
import torch
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time

import copy
from prettytable import PrettyTable
import numpy as np
from utils.fed_utils import average_weights, average_weights_EMA, count_parameters


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    cfg.OPTIM.LR = args.lr # learning rate
    # If True, tfm_train and tfm_test will be None, only use range(0,1) & normalize
    cfg.INPUT.NO_TRANSFORM = args.input_no_transform
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.STEPSIZE = args.stepsize

def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for GLP_OT
    cfg.TRAINER.GLP_OT = CN()
    cfg.TRAINER.GLP_OT.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.GLP_OT.CSC = False  # class-specific context
    cfg.TRAINER.GLP_OT.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.GLP_OT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GLP_OT.N = args.num_prompt # number of prompts
    cfg.TRAINER.GLP_OT.THRESH = args.thresh # thresh of sinkhorn distance
    cfg.TRAINER.GLP_OT.EPS = args.eps # lambada of sinkhorn distance
    cfg.TRAINER.GLP_OT.OT = args.OT # type of OT used
    cfg.TRAINER.GLP_OT.TOP_PERCENT = args.top_percent
    cfg.TRAINER.GLP_OT.MAX_ITER = args.max_iter

    # config for FairLoRA
    cfg.TRAINER.GLP_OT_LORA = CN()
    cfg.TRAINER.GLP_OT_LORA.UNFREEZE_IMAGE_ENCODER = args.unfreeze_image_encoder
    cfg.TRAINER.GLP_OT_LORA.UNFREEZE_TEXT_ENCODER = args.unfreeze_text_encoder
    cfg.TRAINER.GLP_OT_LORA.RANK = args.lora_rank
    cfg.TRAINER.GLP_OT_LORA.ALPHA = args.lora_alpha
    cfg.TRAINER.GLP_OT_LORA.TYPE = args.lora_type
    cfg.TRAINER.GLP_OT_LORA.LOCAL_S = args.lora_local_s
    cfg.TRAINER.GLP_OT_LORA.GLOBAL_S = args.lora_global_s
    cfg.TRAINER.LAMBDA_FAIRNESS = args.lambda_fairness

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clients
    cfg.DATASET.ATTRIBUTE_TYPE = args.attribute_type  # attribute type for medical data, it can be gender, race, ...
    cfg.DATASET.ATTRIBUTES = args.attributes
    cfg.DATASET.MODALITY_TYPE = args.modality_type
    cfg.DATASET.DIM_PER_3D_SLICE = args.dim_per_3d_slice
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = 1 # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step
    cfg.OPTIM.LR = args.lr # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    # global_trainer = build_trainer(cfg)
    # print("type",type(global_trainer))
    # global_trainer.fed_before_train(is_global=True)

    # copy weights
    # global_weights = global_trainer.model.state_dict()
    local_weights= [[] for i in range(args.num_users)]
    local_weights_0= [[] for i in range(args.num_users)]
    local_weights_1= [[] for i in range(args.num_users)]
    local_weights_per = [{} for i in range(args.num_users)]
    local_proj = [{} for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model,"prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    # local_trainers = {net_i: None for net_i in range(cfg.DATASET.USERS)}
    datanumber_client = []
    datanumber_client_by_attr = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(
                len(local_trainer.fed_train_loader_x_dict[net_i].dataset)
            )
            datanumber_client_by_attr.append(
                local_trainer.fed_train_loader_x_dict[net_i].dataset.count_by_attribute(args.attribute_type)
            )
            # local_trainer.fed_before_train()
            # local_trainers[net_i] = local_trainer
            # local_weights[net_i] = copy.deepcopy(local_trainer.model.state_dict())
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_test_auc_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()
    n_cls = len(local_trainer.dm.dataset.classnames)
    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("------------local test start-------------")
            results = []
            # idxs_users = list(range(0, cfg.DATASET.USERS))
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                results.append(local_trainer.test(idx=idx,current_epoch=epoch))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            if cfg.DATASET.NAME == 'DomainNet':
                if epoch == max_epoch-1 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                    print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                    print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                    print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                    print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                    print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            elif cfg.DATASET.NAME == 'Office':
                if epoch == max_epoch-1 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                    print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                    print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                    print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            print("------------local test finish-------------")
            print("Epoch on server :", epoch)
            break

        elif args.model == "fedavg":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights,strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights,idxs_users, datanumber_client)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            for idx in all_users:
                local_trainer.model.load_state_dict(global_weights,strict=False)
                results.append(local_trainer.test(idx=idx,current_epoch=epoch))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            if cfg.DATASET.NAME == 'DomainNet':
                if epoch >= 5 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                    print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                    print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                    print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                    print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                    print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            elif cfg.DATASET.NAME == 'Office':
                if epoch >= 5 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                    print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                    print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                    print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            print("------------local test finish-------------")
            print("Epoch on server :", epoch)

        elif args.model == "fedprox":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights,strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, global_weight=global_weights, fedprox=True, mu=args.mu)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)
            # update global weights
            # global_trainer.model.load_state_dict(global_weights)

            print("------------local test start-------------")
            results = []
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights,strict=False)
                results.append(local_trainer.test(idx=idx,current_epoch=epoch))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            if cfg.DATASET.NAME == 'DomainNet':
                if epoch >= 5 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                    print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                    print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                    print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                    print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                    print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            elif cfg.DATASET.NAME == 'Office':
                if epoch >= 5 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                    print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                    print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                    print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            print("------------local test finish-------------")
            print("Epoch on server :", epoch)

        elif args.model in {'PromptFL', 'FedOTP'}:
            # global prompt + local prompt
            if epoch == 0:
                idxs_users = list(range(0, cfg.DATASET.USERS))
            else:              
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                # gloabl embeddings
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][:args.avg_prompt])  
                # local embeddings
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.avg_prompt:args.num_prompt])
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            
            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx'] = torch.cat([global_weights, local_weights_1[idx]],dim=0)
            if args.num_users >= 50:
                if epoch >= 140:
                    for idx in all_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
                        results.append(local_trainer.test(idx=idx,current_epoch=epoch))
                    global_test_acc = []
                    global_test_error = []
                    global_test_f1 = []
                    for k in range(len(results)):
                        global_test_acc.append(results[k][0])
                        global_test_error.append(results[k][1])
                        global_test_f1.append(results[k][2])
                    global_time_list.append(time.time() - start)
                    global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                    global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                    global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                    global_epoch_list.append(epoch)
                    print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                    print("Global test error:", sum(global_test_error) / len(global_test_error))
                    print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                    print("------------local test finish-------------")
                    print("Epoch on server :", epoch)
            else:
                for idx in all_users:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                    results.append(local_trainer.test(idx=idx,current_epoch=epoch))

                global_test_acc = []
                global_test_error = []
                global_test_f1 = []
                global_test_auc = []
                for k in range(len(results)):
                    global_test_acc.append(results[k][0])   # "accuracy"
                    global_test_error.append(results[k][1]) # "error_rate"
                    global_test_f1.append(results[k][2 ])   # "macro_f1"
                    if len(results[k]) > 3:
                        global_test_auc.append(results[k][3])  # "auc"
                global_time_list.append(time.time() - start)
                global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    global_test_auc_list.append(sum(global_test_auc) / len(global_test_auc))
                global_epoch_list.append(epoch)
                print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                print("Global test error:", sum(global_test_error) / len(global_test_error))
                print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    print("Global test auc:", sum(global_test_auc) / len(global_test_auc))
                if cfg.DATASET.NAME == 'DomainNet':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                        print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                        print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                        print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                        print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                        print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == 'Office':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                        print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                        print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                        print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == "HarvardOph":
                    pass
                print("------------local test finish-------------")
                print("Epoch on server :", epoch)
                print()
        
        elif args.model == 'FedOTPLinearFT':  
            # global prompt + local prompt + lora on image encoder
            if epoch == 0:
                idxs_users = list(range(0, cfg.DATASET.USERS))
            else:              
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.avg_prompt:args.num_prompt])
                local_weights_1[idx] = copy.deepcopy({k:v for k,v in local_weight.items() if 'lora_S' in k})
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)
            # update global weights
            # global_trainer.model.load_state_dict(global_weights)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            
            for idx in all_users:
                local_weights_per[idx] = copy.deepcopy(global_weights)
                # keep local embeddings
                local_weights_per[idx]['prompt_learner.ctx'][args.avg_prompt:args.num_prompt] = local_weights_0[idx]
                if cfg.TRAINER.GLP_OT_LORA.LOCAL_S:
                    for k, v in local_weights_1[idx].items():
                        local_weights_per[idx][k] = v
            if args.num_users >= 50:
                if epoch >= 140:
                    for idx in all_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
                        results.append(local_trainer.test(idx=idx,current_epoch=epoch))
                    global_test_acc = []
                    global_test_error = []
                    global_test_f1 = []
                    for k in range(len(results)):
                        global_test_acc.append(results[k][0])
                        global_test_error.append(results[k][1])
                        global_test_f1.append(results[k][2])
                    global_time_list.append(time.time() - start)
                    global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                    global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                    global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                    global_epoch_list.append(epoch)
                    print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                    print("Global test error:", sum(global_test_error) / len(global_test_error))
                    print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                    print("------------local test finish-------------")
                    print("Epoch on server :", epoch)
            else:
                for idx in all_users:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                    results.append(local_trainer.test(idx=idx,current_epoch=epoch))

                global_test_acc = []
                global_test_error = []
                global_test_f1 = []
                global_test_auc = []
                for k in range(len(results)):
                    global_test_acc.append(results[k][0])   # "accuracy"
                    global_test_error.append(results[k][1]) # "error_rate"
                    global_test_f1.append(results[k][2 ])   # "macro_f1"
                    if len(results[k]) > 3:
                        global_test_auc.append(results[k][3])  # "auc"
                global_time_list.append(time.time() - start)
                global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    global_test_auc_list.append(sum(global_test_auc) / len(global_test_auc))
                global_epoch_list.append(epoch)
                print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                print("Global test error:", sum(global_test_error) / len(global_test_error))
                print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    print("Global test auc:", sum(global_test_auc) / len(global_test_auc))
                if cfg.DATASET.NAME == 'DomainNet':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                        print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                        print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                        print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                        print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                        print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == 'Office':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                        print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                        print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                        print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == "HarvardOph":
                    pass
                print("------------local test finish-------------")
                print("Epoch on server :", epoch)
                print()
        
        elif args.model == 'FedOTPLoRA': # fairlora is based on FedOPT framework
            # global prompt + local prompt + lora on image encoder
            if epoch == 0:
                idxs_users = list(range(0, cfg.DATASET.USERS))
            else:              
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)

                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, is_last_client=idx==idxs_users[-1])
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.avg_prompt:args.num_prompt])
                local_weights_1[idx] = copy.deepcopy({k:v for k,v in local_weight.items() if 'lora_S' in k})
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            # Important!!
            print('Use EMA')
            global_weights = average_weights_EMA(global_weights, local_weights, idxs_users, 
                                                 datanumber_client, datanumber_client_by_attr, 
                                                 epoch, max_epoch, shared_half_s=args.shared_half_s)
            # global_weights = average_weights(local_weights, idxs_users, datanumber_client, datanumber_client_by_attr)
            # update global weights
            # global_trainer.model.load_state_dict(global_weights)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            
            for idx in all_users:
                local_weights_per[idx] = copy.deepcopy(global_weights)
                # keep local embeddings
                local_weights_per[idx]['prompt_learner.ctx'][args.avg_prompt:args.num_prompt] = local_weights_0[idx]
                if cfg.TRAINER.GLP_OT_LORA.LOCAL_S:
                    for k, v in local_weights_1[idx].items():
                        local_weights_per[idx][k] = v

            if args.num_users >= 50:
                if epoch >= 140:
                    for idx in all_users:
                        local_trainer.model.load_state_dict(local_weights_per[idx],strict=False)
                        results.append(local_trainer.test(idx=idx,current_epoch=epoch))
                    global_test_acc = []
                    global_test_error = []
                    global_test_f1 = []
                    for k in range(len(results)):
                        global_test_acc.append(results[k][0])
                        global_test_error.append(results[k][1])
                        global_test_f1.append(results[k][2])
                    global_time_list.append(time.time() - start)
                    global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                    global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                    global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                    global_epoch_list.append(epoch)
                    print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                    print("Global test error:", sum(global_test_error) / len(global_test_error))
                    print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                    print("------------local test finish-------------")
                    print("Epoch on server :", epoch)
            else:
                for idx in all_users:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                    results.append(local_trainer.test(idx=idx, current_epoch=epoch))

                global_test_acc = []
                global_test_error = []
                global_test_f1 = []
                global_test_auc = []
                for k in range(len(results)):
                    global_test_acc.append(results[k][0])   # "accuracy"
                    global_test_error.append(results[k][1]) # "error_rate"
                    global_test_f1.append(results[k][2 ])   # "macro_f1"
                    if len(results[k]) > 3:
                        global_test_auc.append(results[k][3])  # "auc"
                global_time_list.append(time.time() - start)
                global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
                global_test_error_list.append(sum(global_test_error) / len(global_test_error))
                global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    global_test_auc_list.append(sum(global_test_auc) / len(global_test_auc))
                global_epoch_list.append(epoch)
                print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
                print("Global test error:", sum(global_test_error) / len(global_test_error))
                print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
                if len(results[k]) > 3:
                    print("Global test auc:", sum(global_test_auc) / len(global_test_auc))

                if cfg.DATASET.NAME == 'DomainNet':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                        print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                        print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                        print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                        print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                        print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == 'Office':
                    if epoch >= 5 and args.split_client:
                        print("Test acc of clients:", global_test_acc)
                        print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                        print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                        print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                        print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                        print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
                elif cfg.DATASET.NAME == "HarvardOph":
                    pass
                print("------------local test finish-------------")
                print("Epoch on server :", epoch)
                print()

        elif args.model == "local":
            # idxs_users = list(range(0,cfg.DATASET.USERS))
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            results = []
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                results.append(local_trainer.test(idx=idx,current_epoch=epoch))
            global_test_acc = []
            global_test_error = []
            global_test_f1 = []
            for k in range(len(results)):
                global_test_acc.append(results[k][0])
                global_test_error.append(results[k][1])
                global_test_f1.append(results[k][2])
            global_time_list.append(time.time() - start)
            global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
            global_test_error_list.append(sum(global_test_error) / len(global_test_error))
            global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
            global_epoch_list.append(epoch)
            print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
            print("Global test error:", sum(global_test_error) / len(global_test_error))
            print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
            if cfg.DATASET.NAME == 'DomainNet':
                if epoch == max_epoch-1 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of clipart", np.mean(global_test_acc[0:5]),"±",np.std(global_test_acc[0:5]))
                    print("Test acc of infograph", np.mean(global_test_acc[5:10]),"±",np.std(global_test_acc[5:10]))
                    print("Test acc of painting", np.mean(global_test_acc[10:15]),"±",np.std(global_test_acc[10:15]))
                    print("Test acc of quickdraw", np.mean(global_test_acc[15:20]),"±",np.std(global_test_acc[15:20]))
                    print("Test acc of real", np.mean(global_test_acc[20:25]),"±",np.std(global_test_acc[20:25]))
                    print("Test acc of sketch", np.mean(global_test_acc[25:30]),"±",np.std(global_test_acc[25:30]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            elif cfg.DATASET.NAME == 'Office':
                if epoch == max_epoch-1 and args.split_client:
                    print("Test acc of clients:", global_test_acc)
                    print("Test acc of amazon", np.mean(global_test_acc[0:3]),"±",np.std(global_test_acc[0:3]))
                    print("Test acc of caltech", np.mean(global_test_acc[3:6]),"±",np.std(global_test_acc[3:6]))
                    print("Test acc of dslr", np.mean(global_test_acc[6:9]),"±",np.std(global_test_acc[6:9]))
                    print("Test acc of webcam", np.mean(global_test_acc[9:12]),"±",np.std(global_test_acc[9:12]))
                    print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
            print("------------local test finish-------------")
            break

    for idx in all_users:
        filename = os.path.join(args.output_dir, f'global_client{idx}_final.pth') 
        print(f"Save client-{idx} global weights: {filename}")
        torch.save(local_weights_per[idx], filename)

    for idx in idxs_users:
        local_trainer.fed_after_train()
    # global_trainer.fed_after_train()
    print("global_test_acc_list:",global_test_acc_list)
    print("maximum test acc:", max(global_test_acc_list))
    print("mean of acc:",np.mean(global_test_acc_list[-5:]))
    print("std of acc:",np.std(global_test_acc_list[-5:]))
    if len(global_test_auc_list):
        print("global_test_auc_list:", global_test_auc_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FedOTP", help="model of aggregation, choose from:FedOTP(used with GLP_OT), fedavg, fedprox, local(The last three are used with PromptFL)")
    parser.add_argument("--trainer", type=str, default="GLP_OT", help="name of trainer, choose from: CLIP, PromptFL, GLP_OT")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--stepsize', type=int, default=-1, help="step size for lr decay")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num_shots', type=int, default=2, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir100',help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.1,help='The parameter for the dirichlet distribution for data partitioning')
    # domainnet, office
    parser.add_argument('--imbalance_train', default=False, help="is adding label skew to feature skew datasets")
    parser.add_argument('--split_client', default=False, help="is adding label skew to feature skew datasets and split one domain to multi clients")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")
    # attribute
    parser.add_argument('--attribute_type', type=str, default='race',help='the attribute of data used in medical data, it can be gender, race, langugae, ...')
    parser.add_argument('--attributes', type=list, default=['gender', 'race', 'ethnicity', 'language', 'maritalstatus'], help='the data attributes in medical data')
    parser.add_argument('--modality_type', type=str, default='slo_fundus', help='slo_fundus, oct_bscans')
    parser.add_argument('--dim_per_3d_slice', type=int, default=16, help='split oct_bscans into multuple slices, dim of each slice')
    parser.add_argument('--input_no_transform', type=bool, default=False, help='If True, tfm_train and tfm_test will be None.')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=2, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to aggregate")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")

    # parameters of OT
    parser.add_argument('--OT', type=str, default='COT', help="type of OT used: Sinkhorn(for standard OT), COT(for unbalanced OT)")
    parser.add_argument('--top_percent', type=float, default=1, help='the top_percent of COT, control the mapping size of prompts on the feature map')
    parser.add_argument('--eps', type=float, default=0.1, help='the lambada of sinkhorn distance')
    parser.add_argument('--thresh', type=float, default=1e-3, help='the thresh of sinkhorn distance')
    parser.add_argument('--max_iter', type=int, default=100, help="max iteration of COT")

    parser.add_argument('--unfreeze_image_encoder', type=bool, default=False, help='Unfreeze image encoder of CLIP')
    parser.add_argument('--unfreeze_text_encoder', type=bool, default=False, help='Unfreeze text encoder of CLIP')
    parser.add_argument('--lora_rank', type=int, default=4, help='Low rank adaptation')
    parser.add_argument('--lora_alpha', type=float, default=0.04, help='scale = alpha/rank, W_new = W + scale *BA')
    parser.add_argument('--lora_type', type=str, default='LoRA', help='LoRA type: LoRA or SVLoRA, where SVLoRA combines compact SVD and LoRA!')
    parser.add_argument('--lora_local_s', type=bool, default=False, 
                        help='if True, sigular values are viewed as local weights, which DONOT comminicate with golbal weights')
    parser.add_argument('--shared_half_s', type=bool, default=False, 
                        help='shared half singular values across all groups => higher group fairness')
    parser.add_argument('--lora_global_s', type=bool, default=False, 
        help='if True, we use a set of global sigular values, which is used to comminicate with golbal weights')
    parser.add_argument('--lambda_fairness', type=float, default=0.0, help='loss = cls_loss + lambda_fairness * fairness_loss')

    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="/DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/..", help="output directory")
    parser.add_argument("--config-file", type=str, default="configs/trainers/GLP_OT/rn50.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/caltech101.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()
    main(args)
