#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2
ATTRIBUTE_TYPE=$1 # language  # race, language, ethnicity
echo "Running with ATTRIBUTE_TYPE=$ATTRIBUTE_TYPE"

CFG=$2 # vit_b16_oph  # config file rn50_oph or vit_b16_oph
echo "Running with CFG=$CFG"

# custom config
DATA="DATA/"
MODEL=FedOTP
TRAINER=GLP_OT
PRETRAINED=True
OT=COT
TOP_PERCENT=0.8
EPS=0.1
THRESH=0.001
MAX_ITER=100
LR=0.001
GAMMA=1
USERS=2
FRAC=1.0  # frac * num_users will be trained in per epoch
ROUND=50
STEPSIZE=40
NUM_PROMPT=2
# DATASET=$1
# CFG=rn50_oph  # config file
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
# ATTRIBUTE_TYPE='ethnicity'
# Generate a timestamp in the format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")
for DATASET in fedchexmimic
do
  for PARTITION in noniid-labeldir100
  do
    for SEED in 1
    do
      DIR=output/fedchexmimic/FedOTP_${CFG}/${DATASET}_${PARTITION}_beta${BETA}_normalize/${MODEL}_${TRAINER}_${OT}_${TOP_PERCENT}_eps${EPS}_${ATTRIBUTE_TYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}_${timestamp}
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
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GLP_OT/${CFG}.yaml \
        --output-dir ${DIR} \
        --attributes race gender age
      fi
    done
  done
done