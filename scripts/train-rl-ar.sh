#!/usr/bin/env bash

RUN_NAME="rl-ar-var-20pnn-gnn-max"

PROBLEM="tsp"

DEVICES="0"
NUM_WORKERS=0

MIN_SIZE=20
MAX_SIZE=50
NEIGHBORS=0.2
KNN_STRAT="percentage"

DISTRIBUTION="random"

VAL_DATASET1="data/tsp/tsp20_test_concorde.txt"
VAL_DATASET2="data/tsp/tsp50_test_concorde.txt"
# VAL_DATASET3="data/tsp/tsp100_test_concorde.txt"

N_EPOCHS=100
EPOCH_SIZE=128000
BATCH_SIZE=128
ACCUMULATION_STEPS=1

VAL_SIZE=1280
ROLLOUT_SIZE=1280

MODEL="attention"
ENCODER="gnn"
AGGREGATION="max"
AGGREGATION_GRAPH="mean"
NORMALIZATION="batch"
EMBEDDING_DIM=128
N_ENCODE_LAYERS=3

BASELINE="rollout"

LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" --baseline "$BASELINE" \
    --min_size "$MIN_SIZE" --max_size "$MAX_SIZE" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --data_distribution "$DISTRIBUTION" \
    --val_datasets "$VAL_DATASET1" "$VAL_DATASET2" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --val_size "$VAL_SIZE" --rollout_size "$ROLLOUT_SIZE" \
    --encoder "$ENCODER" --aggregation "$AGGREGATION" --aggregation_graph "$AGGREGATION_GRAPH" \
    --n_encode_layers "$N_ENCODE_LAYERS" --gated \
    --normalization "$NORMALIZATION" --learn_norm \
    --embedding_dim "$EMBEDDING_DIM" --hidden_dim "$EMBEDDING_DIM" \
    --lr_model "$LR_MODEL" --max_grad_norm "$MAX_NORM" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
    --run_name "$RUN_NAME"