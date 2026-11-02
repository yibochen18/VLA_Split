#!/usr/bin/env bash
# Real MLP on CV folds 1-4, run like the linear heads so the scenes match.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_head/mlp_refs

log() { echo "=== [$(date '+%F %T')] $* ==="; }

# Wait for room on the shared GPU; retry failed starts
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
gpu_run() {
    for attempt in $(seq 12); do
        wait_gpu
        if "$@"; then return 0; fi
        log "attempt $attempt failed, retrying in 2 min"; sleep 120
    done
    log "GIVING UP: $*"; return 1
}

mlp_run() {  # suite fold
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$1" --first_episode $((10 * $2)) \
        --episodes_per_task 10 --bundle "$VLA_SPLIT/$1/model2" --tag "mlp_fold$2" --split_root "$ROOT" \
        2>&1 | grep -E '^Success|Traceback|Error')
}

# Wait for the depth sweep
until grep -q 'ALL DONE' "$VLA_LOGS/linear_layers.log" 2>/dev/null; do sleep 300; done

for suite in libero_spatial libero_10 libero_goal libero_object; do
    for f in 1 2 3 4; do
        [ $suite = libero_spatial ] && [ $f = 4 ] && continue
        log "real MLP $suite fold $f"
        gpu_run mlp_run $suite $f
    done
done
log "ALL DONE"
