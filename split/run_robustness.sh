#!/usr/bin/env bash
# Real model vs linear head vs layer-20 replacement under each perturbation.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/robustness
mkdir -p "$ROOT"

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

eval_run() {  # suite perturbation policy
    local bundle extra=()
    case $3 in
        real) bundle=$VLA_SPLIT/$1/model2 ;;
        linear) bundle=$VLA_DATA/linear_head/heads/$1_fold4 ;;
        L20) bundle=$VLA_DATA/linear_layers/heads/$1_L20; extra=(--interface_layer 20 --truncate) ;;
    esac
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$1" --first_episode 40 \
        --episodes_per_task 10 --bundle "$bundle" --tag "$2_$3" --split_root "$ROOT/evals" --perturb "$2" \
        "${extra[@]}" 2>&1 | grep -E '^Success|Traceback|Error')
}


for suite in libero_spatial libero_10; do
    for kind in camera objects dark noise; do
        for policy in real linear L20; do
            log "$suite $kind $policy"
            gpu_run eval_run "$suite" "$kind" "$policy"
        done
    done
done
log "ALL DONE"
