#!/usr/bin/env bash
# Layer-depth extraction test: copy Model 2 from layers 28/30/31 (cuts B-4/B-2/B-1), Spatial.
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_EXTRACTION/libero_spatial_layers
SPLIT_ROOT=$VLA_DATA/split_layers
LAYERS="28 30 31"

cd "$VLA_ROOT/openvla-oft"
echo "=== [$(date '+%F %T')] recording layers $LAYERS (500 episodes, real Model 2) ==="
python ../split/run_split_eval.py --suite libero_spatial --split_root "$SPLIT_ROOT" \
    --bundle "$VLA_SPLIT/libero_spatial/model2" \
    --save_pairs "$ROOT/pairs" --record_layers $LAYERS 2>&1 | grep -E '^Success|Traceback|Error'

cd "$VLA_ROOT/split"
for layer in $LAYERS; do
    for arch in linear mlp oft; do
        for n in 100 500 1000 2500 5000; do
            python train_replacement.py --arch $arch --num_queries $n \
                --pairs_dir "$ROOT/pairs/L$layer" --out_root "$ROOT/replacements/L$layer" | sed "s/^/L$layer /"
        done
    done
done

# Closed loop on a subset of N
cd "$VLA_ROOT/openvla-oft"
for n in 100 1000 5000; do
    for layer in $LAYERS; do
        for arch in oft mlp linear; do
            echo "=== [$(date '+%F %T')] closed loop: L$layer $arch n=$n ==="
            python ../split/run_split_eval.py --suite libero_spatial --first_episode 40 --episodes_per_task 10 \
                --interface_layer $layer --bundle "$ROOT/replacements/L$layer/${arch}_n${n}_s0" \
                --tag L${layer}_${arch}_n${n} --split_root "$SPLIT_ROOT" 2>&1 | grep -E '^Success|Traceback|Error'
        done
    done
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
