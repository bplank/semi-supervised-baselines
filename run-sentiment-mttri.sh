#!/usr/bin/env bash

# paths
TASK=sentiment
SOURCE=books
TARGET=dvd
DATA_DIR=data
EXP_DIR=expdir  # experiment directory
MODELS_DIR=${EXP_DIR}/models
LOGS_DIR=${EXP_DIR}/logs
PRED_DIR=${EXP_DIR}/predictions
BASE_ID=b_asymm_k3060_${TARGET}-sentiment
CONFIG=config/sentiment_mttri.config

# settings
DYNET_MEM=1000

# hyperparameters
NUM_RUNS=1
MAX_VOCAB_SIZE=5000
SIZE_BOOTSTRAP=0.0
ORTHOGONALITY_WEIGHT=0.01

# train base model
python src/ssl_base.py --dynet-autobatch 1\
                       --dynet-mem ${DYNET_MEM}\
                       -d ${DATA_DIR}\
                       -m ${MODELS_DIR}/${BASE_ID}\
                       --task ${TASK}\
                       -s ${SOURCE}\
                       -t ${TARGET}\
                       -l ${LOGS_DIR}/${BASE_ID}/log-base-${TARGET}.txt\
                       --strategy mttri_base\
                       --asymmetric\
                       --num-runs ${NUM_RUNS}\
                       --config ${CONFIG}\
                       --max-vocab-size ${MAX_VOCAB_SIZE}\
                       --save-epoch-1\
                       --output-predictions ${PRED_DIR}/${BASE_ID}\
                       --size-bootstrap ${SIZE_BOOTSTRAP}\
                       --orthogonality-weight ${ORTHOGONALITY_WEIGHT}\

# mt-tri training settings
CONF_THRES=0.9
MT_TRI_ID=mttt_save_final_k3060_${TARGET}-sentiment
ADV_WEIGHT=1.0

# run MTTri
python src/ssl_base.py --dynet-autobatch 1\
                       --dynet-mem ${DYNET_MEM}\
                       -d ${DATA_DIR}\
                       -m ${MODELS_DIR}/${MT_TRI_ID}\
                       --task ${TASK}\
                       -s ${SOURCE}\
                       -t ${TARGET}\
                       -l ${LOGS_DIR}/${MT_TRI_ID}/log-base-${TARGET}.txt\
                       --strategy mttri\
                       --asymmetric\
                       --num-runs ${NUM_RUNS}\
                       --config ${CONFIG}\
                       --max-vocab-size ${MAX_VOCAB_SIZE}\
                       --output-predictions ${PRED_DIR}/${MT_TRI_ID}\
                       --size-bootstrap ${SIZE_BOOTSTRAP}\
                       --start-model-dir ${MODELS_DIR}/${BASE_ID}\
                       --start patience_2\
                       --predict majority\
                       --adversarial-weight ${ADV_WEIGHT}\
                       --orthogonality-weight ${ORTHOGONALITY_WEIGHT}\
                       --asymmetric-type pair\
                       --candidate-pool-scheduling\
                       --confidence-threshold ${CONF_THRES}\
                       --save-final-model\
