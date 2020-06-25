#!/usr/bin/env bash

FT_RUN_NAME="active-exp-lr1e-5"
FT_STRATEGY="active"

DEVICES="0"

VAL_DATASET="data/tsp/tsp200_test_concorde.txt"
EPOCH_SIZE=12800
VAL_SIZE=1280

BATCH_SIZE=16
ACCUMULATION_STEPS=8
N_EPOCHS=100
VAL_EVERY=1

MODEL="outputs/tsp_20-50/rl-ar-var-20pnn-gnn-max-ln-rerun_20200427T123442"

BASELINE="exponential"

LR_FT=0.00001
MAX_NORM=1


CUDA_VISIBLE_DEVICES="$DEVICES" python finetune.py  \
    --ft_run_name "$FT_RUN_NAME" \
    --ft_strategy "$FT_STRATEGY" \
    --val_dataset "$VAL_DATASET" \
    --epoch_size "$EPOCH_SIZE" --val_every "$VAL_EVERY" \
    --val_size "$VAL_SIZE" --rollout_size "$VAL_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --model "$MODEL" \
    --baseline "$BASELINE" \
    --lr_ft "$LR_FT" --max_grad_norm "$MAX_NORM" 
