DEVICES="0"
NUM_WORKERS=0

EVAL_DATASET="data/tsp/tsp10-200_concorde.txt"

VAL_SIZE=25600

MODELS=("outputs/tsp_20-50/rl-ar-var-20pnn-gnn-max_20200313T002243")

BATCH_SIZE=16

for MODEL in ${MODELS[*]}; do
    echo $MODEL
    CUDA_VISIBLE_DEVICES="$DEVICES" python eval.py  \
        "$EVAL_DATASET" \
        --val_size "$VAL_SIZE" --batch_size "$BATCH_SIZE" \
        --model "$MODEL" \
        --decode_strategies "greedy" "bs" \
        --widths 0 128 \
        --num_workers "$NUM_WORKERS"
done

# For insertion baselines:
# python eval_baseline.py random_insertion data/tsp/tsp10-200_concorde.txt -n 25600 --cpus 32 -f
