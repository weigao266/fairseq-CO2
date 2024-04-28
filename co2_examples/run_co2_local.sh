#! /usr/bin/bash

# export PATH=/usr/local/cuda-11.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

BATCH_SIZE=2
TOKENS_PER_SAMPLE=2048
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))
DATA_DIR=/cpfs01/user/sunweigao/wikitext-opt
ARCH=transformer_lm_gpt2_medium

REMARK=CO2
rm -rf checkpoints

prefix=lm
MAX_UPDATE=100000
WARM_UP=2000
# UPDATE_FREQ=$(( 128 / $BATCH_SIZE / 8 ))
UPDATE_FREQ=1
PORT=$(( $RANDOM + 2000 ))
echo $PORT
LR=0.0005
CLIP_NORM=1.0
decay=0.2

logger_dir=./logs
mkdir -p $logger_dir
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=${logger_dir}/${ARCH}_${START_TIME}_$REMARK.log
TENSOR_DIR=tensorboard/tensorboard-co2/$START_TIME-$ARCH-$REMARK

torchrun --standalone --nproc_per_node=4 \
    $(which fairseq-train) --task language_modeling \
        $DATA_DIR --bpe gpt2 \
        --save-dir checkpoints/$prefix/${ARCH} \
        --arch $ARCH --clip-norm=$CLIP_NORM \
        --ddp-backend co2 --co2-outer-momentum 0.2 --co2-base-algorithm localsgd --co2-clip --co2-clip-threshold 1.0 --co2-use-streams \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 128 --min-loss-scale 0.0001220703125 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay $decay \
        --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-08 \
        --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
        --update-freq $UPDATE_FREQ \
        --batch-size $BATCH_SIZE --num-workers 4 \
        --max-update $MAX_UPDATE --log-format json --log-interval 1 2>&1 | sudo tee -a $LOG_FILE


# --ddp-backend co2 --co2-outer-momentum 0.2 --co2-base-algorithm localsgd --co2-clip --co2-clip-threshold 1.0 \
