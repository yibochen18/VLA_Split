#!/usr/bin/env bash
# Extraction test (cut A, Spatial): train replacement Model 2s, score them on starts 40-49.
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_EXTRACTION/libero_spatial

# Wait for the pair-collection run's summary (pgrep would match the tmux command line)
until [ -f "$VLA_SPLIT"/libero_spatial/split_eval_seed7_50ep.json ]; do sleep 60; done
echo "=== [$(date '+%F %T')] pairs: $(ls $ROOT/pairs | wc -l) episodes ==="

cd "$VLA_ROOT/split"
for arch in linear mlp oft; do
    for n in 100 500 1000 2500 5000; do
        python train_replacement.py --arch $arch --num_queries $n
    done
done

cd "$VLA_ROOT/openvla-oft"
for arch in oft mlp linear; do
    for n in 100 500 1000 2500 5000; do
        echo "=== [$(date '+%F %T')] closed loop: $arch n=$n ==="
        python ../split/run_split_eval.py --suite libero_spatial --first_episode 40 --episodes_per_task 10 \
            --bundle $ROOT/replacements/${arch}_n${n}_s0 --tag ${arch}_n${n} --split_root "$VLA_SPLIT" \
            2>&1 | grep -E '^Success|Traceback|Error'
    done
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
