#!/bin/bash
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
BASE_DATA_PATH=/home/ubuntu/training_datasets
DATASET=${BASE_DATA_PATH}/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

# You can change the ZeRO stage to 1, 2, or 3
TP=1
PP=1
DP=1
SP=1
WORLD_SIZE=$((TP*PP*DP*SP))

ZERO_STAGE=3
CPU_OFFLOAD="true"
ACTIVATION_CHECKPOINTING="true"
MICRO_BATCH=2
GLOBAL_BATCH=$((MICRO_BATCH*WORLD_SIZE))

MODEL_SIZE="gpt3_20B_v1"
EXIT_INTERVAL=20
DTYPE="fp16"
SEQ=4096

if [[ $MODEL_SIZE == "gpt3_1.3B" ]]; then
        HIDDEN=2048
        LAYERS=24
        ATTENTION_HEADS=16
        SIZE_TAG="gpt3_1.3B"
elif [[ $MODEL_SIZE == "gpt3_1.7B" ]]; then
        HIDDEN=2048
        LAYERS=20
        ATTENTION_HEADS=16
        SIZE_TAG="gpt3_1.7B"
elif [[ $MODEL_SIZE == "gpt3_3B" ]]; then
        HIDDEN=4096
        LAYERS=16
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_3B"
elif [[ $MODEL_SIZE == "gpt3_5B" ]]; then
        HIDDEN=4096
        LAYERS=23
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_5B"
elif [[ $MODEL_SIZE == "gpt3_6B" ]]; then
        HIDDEN=4096
        LAYERS=28
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_6B"
elif [[ $MODEL_SIZE == "gpt3_6.7B" ]]; then
        HIDDEN=4096
        LAYERS=32
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_6.7B"
elif [[ $MODEL_SIZE == "gpt3_13B" ]]; then
        HIDDEN=4096
        LAYERS=60
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_13B"
elif [[ $MODEL_SIZE == "gpt3_20B_v1" ]]; then
        HIDDEN=8192
        LAYERS=24
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_20B"
elif [[ $MODEL_SIZE == "gpt3_20B" ]]; then
        HIDDEN=6144
        LAYERS=44
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_20B"
elif [[ $MODEL_SIZE == "gpt3_30B" ]]; then
        HIDDEN=7168
        LAYERS=48
        ATTENTION_HEADS=56
        SIZE_TAG="gpt3_30B"
elif [[ $MODEL_SIZE == "gpt3_70B" ]]; then
        HIDDEN=8192
        LAYERS=80
        ATTENTION_HEADS=64
        SIZE_TAG="gpt3_70B"
else
        HIDDEN=5120
        LAYERS=40
        ATTENTION_HEADS=80
        SIZE_TAG="gpt3_13B"
fi

# 3D parallelism of training 
TRAIN_ITERS=100000
LR=5.0e-5
MIN_LR=1.e-5

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP
RUN_TAG="save"
# RUN_TAG="ref_load${LOADTP}${LOADPP}${LOAD_DP}"
EXP_DIR="z${ZERO_STAGE}_uni_ckpt" 
CHECKPOINT_PATH=${EXP_DIR}/checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}sp${SP}${SIZE_TAG}
LOAD_CHECKPOINT_PATH=${EXP_DIR}/fake_checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}sp${SP}${SIZE_TAG}
LOG_DIR="${EXP_DIR}/tensorboard/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}LR${LR}_${MINLR}${DTYPE}_${SIZETAG}${RUN_TAG}"
mkdir -p $LOG_DIR

options=" \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
        --ds-sequence-parallel-size $SP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads $ATTENTION_HEADS \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITERS \
        --lr $LR \
    --min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 400 \
        --eval-interval 200 \
    --data-path ${DATASET} \
    --vocab-file ${VOCAB_PATH} \
    --merge-file ${MERGE_PATH} \
    --save-interval 100 \
        --split 98,2,0 \
        --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
        --${DTYPE} \
    --exit-interval ${EXIT_INTERVAL} \
        --save ${CHECKPOINT_PATH} \
        --load ${LOAD_CHECKPOINT_PATH} \
        --make-vocab-size-divisible-by 256 \
    --tensorboard-dir $LOG_DIR
        "

if [ "${ACTIVATION_CHECKPOINTING}" = "true" ]; then
options="${options} \
        --checkpoint-activations \
        --deepspeed-activation-checkpointing"
fi

if [ "${CPU_OFFLOAD}" = "true" ]; then
options="${options} \
        --cpu-optimizer"
fi

options="${options} \
        --deepspeed \
        --deepspeed_config=${CONFIG_JSON} \
        --zero-stage=${ZERO_STAGE}
"

if [[ ${ZERO_STAGE} -gt 1 ]]; then
options="${options} \
    --no-pipeline-parallel"
fi

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": false,
    "reduce_bucket_size": 8e8,
    "sub_group_size" : 8e8$(if [ "${CPU_OFFLOAD}" = "true" ]; then echo ',
    "contiguous_gradients" : false,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "ratio": 0.98
    }'; fi)
  },
  "bf16": {
    "enabled": false
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 50,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
run_cmd="deepspeed $WORKER_STR ${DIR}/pretrain_gpt.py ${options}"
# run_cmd="deepspeed --bind_cores_to_rank --master_port 29700 $WORKER_STR ${DIR}/pretrain_gpt.py ${options}"
echo ${options}
echo ${run_cmd}
eval ${run_cmd}

set +x