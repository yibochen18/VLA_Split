#!/usr/bin/env bash
# Long, layer-24 readout replicated on starts 30-39 (RESULTS §12).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/linear_followup
suite=libero_10; L=24
log() { echo "=== [$(date '+%F %T')] $* ==="; }

log "fit $suite L$L (test 30-39)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$VLA_ROOT/split/ridge_head.py" \
    --pairs_dir "$VLA_DATA/linear_layers/pairs/$suite/L$L" \
    --test_from 30 --test_to 40 --val_from 40 --val_to 50 \
    --out "$ROOT/heads/${suite}_L${L}_test30" 2>&1 | grep -E '^ridge|Traceback|Error'

log "closed loop $suite L$L (starts 30-39)"
(cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$suite" --first_episode 30 \
    --episodes_per_task 10 --bundle "$ROOT/heads/${suite}_L${L}_test30" --tag "L${L}_ridge_test30" \
    --split_root "$ROOT/evals" --interface_layer "$L" --truncate 2>&1 | grep -E '^Success|Traceback|Error')
log "ALL DONE"
