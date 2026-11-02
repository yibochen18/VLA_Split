#!/usr/bin/env bash
# Linear replacement: compute benchmark, truncated closed loops, data efficiency.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_compute
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

eval_run() {  # suite tag bundle [layer]
    local layer_arg=(); [ -n "$4" ] && layer_arg=(--interface_layer "$4" --truncate)
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$1" --first_episode 40 \
        --episodes_per_task 10 --bundle "$3" --tag "$2" --split_root "$ROOT/evals" "${layer_arg[@]}" \
        2>&1 | grep -E '^Success|^Latency|Traceback|Error')
}

bench() {  # suite
    (cd "$VLA_ROOT/openvla-oft" && python ../split/bench_depth.py --suite "$1" --out "$ROOT/bench/$1.json" \
        2>&1 | grep -E '^[0-9]+ |^mlp|^linear|^wrote|Traceback|Error')
}

pairs_dir() {  # suite layer
    case "$1/$2" in
        libero_spatial/32) echo "$VLA_EXTRACTION/libero_spatial/pairs" ;;
        libero_10/32) echo "$VLA_DATA/linear_head/pairs/libero_10" ;;
        *) echo "$VLA_DATA/linear_layers/pairs/$1/L$2" ;;
    esac
}

fit_n() {  # suite layer n (GPU, CPU fallback)
    local a=(--pairs_dir "$(pairs_dir "$1" "$2")" --test_from 40 --test_to 50 --val_from 0 --val_to 10
             --max_queries "$3" --out "$ROOT/heads/$1_L$2_n$3")
    local out
    out=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$VLA_ROOT/split/ridge_head.py" "${a[@]}" 2>&1 \
        | grep -E '^ridge|Traceback|Error'); echo "$out"
    grep -q '^ridge' <<< "$out" && return 0
    log "GPU fit failed, falling back to CPU"
    python "$VLA_ROOT/split/ridge_head.py" --device cpu "${a[@]}" 2>&1 | grep -E '^ridge|Traceback|Error' | grep '^ridge'
}

data_efficiency() {  # suite "layers" "ns"
    for L in $2; do
        for n in $3; do
            log "fit $1 L$L n=$n"
            fit_n "$1" "$L" "$n" || continue
            log "closed loop $1 L$L n=$n"
            if [ "$L" = 32 ]; then gpu_run eval_run "$1" "L32_n$n" "$ROOT/heads/$1_L32_n$n"
            else gpu_run eval_run "$1" "L${L}_n$n" "$ROOT/heads/$1_L${L}_n$n" "$L"; fi
        done
    done
}

if [ "$1" = rerun_missing ]; then  # redo failed fits
    for job in "libero_spatial 32" "libero_spatial 24" "libero_spatial 20" "libero_spatial 16" "libero_10 32" "libero_10 20"; do
        set -- $job
        [ -f "$ROOT/heads/$1_L$2_n1000/head.pt" ] || data_efficiency "$1" "$2" 1000
    done
    log "ALL DONE"; exit
fi

log "1. compute benchmark"
gpu_run bench libero_spatial
gpu_run bench libero_10

log "2. closed loop with the truncated Model 1 (Spatial; the sweep scored L28 97, L24 92, L20 96, L16 95; cut A linear 97)"
gpu_run eval_run libero_spatial L32_full "$VLA_DATA/linear_head/heads/libero_spatial_fold4"
for L in 28 24 20 16; do
    log "truncated closed loop libero_spatial L$L"
    gpu_run eval_run libero_spatial "L${L}_trunc" "$VLA_DATA/linear_layers/heads/libero_spatial_L$L" "$L"
done

log "3. data efficiency"
data_efficiency libero_spatial "32 24 20 16" "100 500 1000 2500"
data_efficiency libero_10 "32 20" "500 1000 2500 5000"

log "ALL DONE"
