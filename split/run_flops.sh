#!/usr/bin/env bash
# FLOPs per query: OFT and π0.5, real vs. readouts (RESULTS §11).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/flops
mkdir -p "$ROOT"

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }

# The suites with a layer-24 linear map ($VLA_DATA/linear_layers/heads/<suite>_L24)
for suite in libero_spatial libero_10; do
    log "OFT $suite"
    wait_gpu
    (cd "$VLA_ROOT/openvla-oft" && python ../split/bench_flops.py --suite "$suite" --out "$ROOT/oft_$suite.json" \
        2>&1 | grep -E '^depth|^mlp|^linear|^real|^wrote|Traceback|Error')
done

log "pi0.5"
wait_gpu
vla_openpi python "$VLA_ROOT/split/bench_flops_pi05.py" --out "$ROOT/pi05.json" 2>&1 \
    | grep -E 'TFLOPs|^wrote|Traceback|Error'
log "done"
