#!/usr/bin/env bash
# Depth sweep: a linear map from layer L replaces the rest; determinism check first.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_layers
SPATIAL_NEW="26 24 22 20 18 16 14 12 10 8 6 4 2"
SPATIAL_ALL="31 30 28 $SPATIAL_NEW"
LONG_LAYERS="28 24 20 16 12 8"

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

eval_run() {  # suite first_episode tag bundle [layer]
    local layer_arg=(); [ -n "$5" ] && layer_arg=(--interface_layer "$5")
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$1" --first_episode "$2" \
        --episodes_per_task 10 --bundle "$4" --tag "$3" --split_root "$ROOT/evals" "${layer_arg[@]}" \
        2>&1 | grep -E '^Success|Traceback|Error')
}

record() {  # suite layers...
    local suite=$1; shift
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$suite" --split_root "$ROOT/record" \
        --bundle "$VLA_SPLIT/$suite/model2" --save_pairs "$ROOT/pairs/$suite" --record_layers "$@" \
        2>&1 | grep -E '^Success|Traceback|Error') || return 1
    # keep only the per-layer pairs
    find "$ROOT/pairs/$suite" -maxdepth 1 -name 'task*_ep*.pt' -delete
}

fit() {  # pairs_dir out
    python "$VLA_ROOT/split/ridge_head.py" --pairs_dir "$1" --test_from 40 --test_to 50 --val_from 0 --val_to 10 \
        --out "$2" 2>&1 | grep -E '^ridge|Traceback|Error' && return 0
    log "GPU fit failed, falling back to CPU"
    python "$VLA_ROOT/split/ridge_head.py" --device cpu --pairs_dir "$1" --test_from 40 --test_to 50 \
        --val_from 0 --val_to 10 --out "$2" 2>&1 | grep -E '^ridge|Traceback|Error'
}

sweep() {  # suite "layers" pairs_dir_for_layer_fn
    local suite=$1 layers=$2 pairs_fn=$3
    for L in $layers; do
        log "fit $suite L$L"
        fit "$($pairs_fn "$suite" "$L")" "$ROOT/heads/${suite}_L$L" || continue
        log "closed loop $suite L$L"
        out=$(gpu_run eval_run "$suite" 40 "L${L}_ridge" "$ROOT/heads/${suite}_L$L" "$L"); echo "$out"
        n=$(echo "$out" | grep -oE 'Success: [0-9]+' | grep -oE '[0-9]+' | tail -1)
        if [ -n "$n" ] && [ "$n" -lt 10 ]; then log "$suite L$L scored $n/100 < 10: skipping deeper layers"; break; fi
    done
}

spatial_pairs() { case $2 in 31|30|28) echo "$VLA_EXTRACTION/libero_spatial_layers/pairs/L$2" ;; *) echo "$ROOT/pairs/$1/L$2" ;; esac; }
long_pairs() { echo "$ROOT/pairs/$1/L$2"; }

log "determinism check: real MLP on Spatial starts 40-49 alone (compare to the full 500-episode run)"
gpu_run bash -c "set -o pipefail; cd '$VLA_ROOT/openvla-oft' && python ../split/run_split_eval.py --suite libero_spatial --first_episode 40 \
    --episodes_per_task 10 --bundle '$VLA_SPLIT/libero_spatial/model2' --tag mlp_fold4_detcheck \
    --split_root '$ROOT/checks' --save_pairs '$ROOT/checks/pairs_mlp_fold4' 2>&1 | grep -E '^Success|Traceback|Error'"

log "record libero_spatial layers $SPATIAL_NEW"
gpu_run record libero_spatial $SPATIAL_NEW
sweep libero_spatial "$SPATIAL_ALL" spatial_pairs

# Long only above the first layer where Spatial scored < 50
spatial_score() { python -c "import json,sys; print(json.load(open(sys.argv[1]))['num_successes'])" \
    "$ROOT/evals/libero_spatial/split_eval_seed7_10ep_from40_L$1_ridge.json" 2>/dev/null || echo 0; }
long_layers=""
for L in $LONG_LAYERS; do
    s=$(spatial_score "$L")
    if [ "$s" -lt 50 ]; then log "Spatial L$L scored $s/100 < 50: not testing Long at L$L or deeper"; break; fi
    long_layers="$long_layers $L"
done
if [ -n "$long_layers" ]; then
    log "record libero_10 layers$long_layers"
    gpu_run record libero_10 $long_layers
    sweep libero_10 "$long_layers" long_pairs
fi

log "ALL DONE"
