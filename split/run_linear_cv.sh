#!/usr/bin/env bash
# 5-fold cross-validation of the ridge linear head over initial states, all suites.
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_DATA/linear_head

# Wait for run_linear_record.sh
until [ -f "$ROOT/record/libero_object/split_eval_seed7_50ep.json" ]; do sleep 60; done

pairs_dir() { [ $1 = libero_spatial ] && echo "$VLA_EXTRACTION/libero_spatial/pairs" || echo "$ROOT/pairs/$1"; }

for suite in libero_spatial libero_10 libero_goal libero_object; do
    for f in 4 0 1 2 3; do
        v=$(( (f + 1) % 5 ))
        head="$ROOT/heads/${suite}_fold$f"
        echo "=== [$(date '+%F %T')] fit $suite fold $f ==="
        python "$VLA_ROOT/split/ridge_head.py" --pairs_dir "$(pairs_dir $suite)" \
            --test_from $((10 * f)) --test_to $((10 * f + 10)) --val_from $((10 * v)) --val_to $((10 * v + 10)) \
            --out "$head" 2>&1 | grep -E '^ridge|Traceback|Error'
        echo "=== [$(date '+%F %T')] closed loop $suite fold $f ==="
        (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite $suite \
            --first_episode $((10 * f)) --episodes_per_task 10 --bundle "$head" \
            --tag ridge_fold$f --split_root "$ROOT/evals" 2>&1 | grep -E '^Success|Traceback|Error')
    done
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
