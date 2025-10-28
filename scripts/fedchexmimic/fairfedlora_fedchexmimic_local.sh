#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
ATTRIBUTE_TYPE=$1 # language  # race, language, ethnicity
echo "Running with ATTRIBUTE_TYPE=$ATTRIBUTE_TYPE"

CFG=$2 #vit_b16_oph  # config file rn50_oph or vit_b16_oph
echo "Running with CFG=$CFG"

idxs_users_train=$3  # 0
idxs_users_test=01
echo "Running with idxs_users_train=$idxs_users_train, idxs_users_test=$idxs_users_test"

# custom config
DATA="DATA/"
MODEL=FedOTPLoRA
TRAINER=GLP_OT_SVLoRA
PRETRAINED=True
OT=None
TOP_PERCENT=0.8
EPS=0.1
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=0.1
USERS=2
FRAC=1.0
ROUND=50
STEPSIZE=200  # 2 * ROUND * 0.5 * max(USERS * FRAC, 1)
NUM_PROMPT=2
# DATASET=$1
# CFG=vit_b16_oph  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CTXINIT=False
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
BETA=0.3
INPUT_NO_TRANSFORM=False
# SEED=1
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
# params for SVLoRA
# ATTRIBUTE_TYPE=language # race, language, ethnicity
UNFREEZE_IMAGE_ENC=True
LoRA_RANK=12
LoRA_ALPHA=2
LoRA_TYPE=FairLoRA
shared_half_s=False
lambda_fairness=0.0
# Generate a timestamp in the format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")
for DATASET in fedchexmimic
do
  for PARTITION in noniid-labeldir100
  do
    for SEED in 1
    do
      DIR=output/fedchexmimic/FairLoRA_${CFG}_ema/train_${idxs_users_train}_test_${idxs_users_test}/${DATASET}_${PARTITION}_beta${BETA}_${LoRA_TYPE}_rank${LoRA_RANK}_alpha${LoRA_ALPHA}_sinit_cycle_shift/${MODEL}_${TRAINER}_${OT}_${TOP_PERCENT}_eps${EPS}_${ATTRIBUTE_TYPE}_lambda_fairness${lambda_fairness}_onehotattr0.7/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}_shared_half_s_${shared_half_s}_${timestamp}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python federated_main.py \
        --root ${DATA} \
        --model ${MODEL} \
        --seed ${SEED} \
        --num_users ${USERS} \
        --frac ${FRAC} \
        --lr ${LR} \
        --OT ${OT} \
        --top_percent ${TOP_PERCENT} \
        --eps ${EPS} \
        --thresh ${THRESH} \
        --max_iter ${MAX_ITER} \
        --gamma ${GAMMA} \
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --stepsize ${STEPSIZE} \
        --input_no_transform ${INPUT_NO_TRANSFORM} \
        --attribute_type ${ATTRIBUTE_TYPE} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --n_ctx ${NCTX} \
        --num_prompt ${NUM_PROMPT} \
        --unfreeze_image_encoder ${UNFREEZE_IMAGE_ENC} \
        --lora_rank ${LoRA_RANK} \
        --lora_alpha ${LoRA_ALPHA} \
        --lora_type ${LoRA_TYPE} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/${CFG}.yaml \
        --output-dir ${DIR} \
        --shared_half_s ${shared_half_s} \
        --lambda_fairness ${lambda_fairness} \
        --attributes race gender age \
        --idxs_users_train ${idxs_users_train} \
        --idxs_users_test ${idxs_users_test}
      fi
    done
  done
done