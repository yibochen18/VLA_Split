#!/usr/bin/env bash
# π0.5 baselines (P4): 500 episodes per suite, recording the backbone states and actions of every query.
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_DATA/pi05

# Wait for room on the shared GPU (π0.5 needs ~8 GB)
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }

cd "$VLA_ROOT/openvla-oft"
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "=== [$(date '+%F %T')] baseline $suite ==="
    wait_gpu
    python ../split/run_pi05_eval.py --suite $suite --out_root "$ROOT/evals" --save_actions \
        --record_dir "$ROOT/record/$suite" 2>&1 | grep -E '^Success|^Latency|Traceback|Error'
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
