#!/bin/bash
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
BASE_DATA_PATH=dataset
DATASET=${BASE_DATA_PATH}/my-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

ZERO_STAGE=1
DTYPE="fp16"
# Debug
DEBUG_MODE=1
if [[ $DEBUG_MODE == 1 ]]; then
        LAYERS=4
        HIDDEN=512
        SEQ=512
        EXIT_INTERVAL=200
        SIZE_TAG="toy"
else
        HIDDEN=1024
        LAYERS=24
        SEQ=1024
        EXIT_INTERVAL=100
        SIZE_TAG="big"
fi  

# 3D parallelism of training 
TP=2
PP=2
DP=2
SP=1
WORLD_SIZE=$((TP*PP*DP*SP))
GLOBAL_BATCH=16
MICRO_BATCH=$((GLOBAL_BATCH/WORLD_SIZE))
TRAIN_ITERS=100000
LR=1.0e-3
MIN_LR=1.0e-4

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -z|--zero-stage)
    ZERO_STAGE=$2
    shift
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    exit 1
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
    --ds-sequence-parallel-size $SP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads 32 \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
        --micro-batch-size $micro_batch \
        --global-batch-size $global_batch \
        --train-iters $TRAIN_ITERS \
        --lr $LR \
        --min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 40 \
        --eval-interval 10 \
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

    options="${options} \
        --deepspeed \
        --deepspeed_config=${CONFIG_JSON} \
        --zero-stage=${ZERO_STAGE} \
        --deepspeed-activation-checkpointing \
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
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": false
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 50,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : false
}
EOT

    WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
    run_cmd="deepspeed --bind_cores_to_rank --master_port 29700 $WORKER_STR ${DIR}/pretrain_gpt.py ${options}"
    
    echo "Running training with micro-batch size: $micro_batch"
    echo ${options}
    echo ${run_cmd}
    eval ${run_cmd}
}

# Main loop to run training for different micro-batch sizes
for micro_batch in $(seq 2 2); do
    echo "Starting training run with micro-batch size: $micro_batch"
    run_training $micro_batch
done