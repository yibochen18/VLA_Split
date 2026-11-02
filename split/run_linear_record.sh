#!/usr/bin/env bash
# Record (interface, action) pairs over all 500 episodes of Long, Goal and Object.
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_DATA/linear_head

cd "$VLA_ROOT/openvla-oft"
for suite in libero_10 libero_goal libero_object; do
    echo "=== [$(date '+%F %T')] recording $suite ==="
    python ../split/run_split_eval.py --suite $suite --split_root "$ROOT/record" \
        --bundle "$VLA_SPLIT/$suite/model2" --save_pairs "$ROOT/pairs/$suite" 2>&1 | grep -E '^Success|Traceback|Error'
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
