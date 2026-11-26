#!/usr/bin/env bash
# Layer-28 readout on Goal and Object (RESULTS §12): record layer-28 pairs, fit, closed loop on starts 40-49.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_followup
L=28
log() { echo "=== [$(date '+%F %T')] $* ==="; }

for suite in libero_goal libero_object; do
    pairs=$ROOT/pairs28/$suite head=$ROOT/heads/${suite}_L$L
    res=$ROOT/evals/$suite/split_eval_seed7_10ep_from40_L${L}_ridge.json
    if [ ! -d "$pairs/L$L" ]; then
        log "record $suite layer $L"
        (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite $suite --split_root "$ROOT/record28" \
            --bundle "$VLA_SPLIT/$suite/model2" --save_pairs "$pairs" --record_layers $L 2>&1 \
            | grep -E '^Success|Traceback|Error') || { log "FAILED record $suite"; continue; }
        find "$pairs" -maxdepth 1 -name 'task*_ep*.pt' -delete   # keep only the per-layer pairs
    fi
    if [ ! -f "$head/config.json" ]; then
        log "fit $suite L$L"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$VLA_ROOT/split/ridge_head.py" --pairs_dir "$pairs/L$L" \
            --test_from 40 --test_to 50 --val_from 0 --val_to 10 --out "$head" 2>&1 | grep -E '^ridge|Traceback|Error'
        [ -f "$head/config.json" ] || { log "FAILED fit $suite"; continue; }
    fi
    [ -f "$res" ] && { log "$suite done already"; continue; }
    log "eval $suite L$L"
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$suite" --first_episode 40 \
        --episodes_per_task 10 --bundle "$head" --tag "L${L}_ridge" --split_root "$ROOT/evals" \
        --interface_layer $L --truncate 2>&1 | grep -E '^Success|Traceback|Error') || log "FAILED eval $suite"
done
log "ALL DONE"
