#!/usr/bin/env bash
# Copying Model 2 on LIBERO-Long (RESULTS §6). Run on GPU 1.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
SUITE=libero_10
PAIRS=$VLA_DATA/linear_head/pairs/$SUITE
ROOT=$VLA_EXTRACTION/$SUITE
log() { echo "=== [$(date '+%F %T')] $* ==="; }
mkdir -p "$ROOT/replacements" "$ROOT/evals"

for n in 100 1000 5000; do
    for arch in oft mlp linear; do
        rep=$ROOT/replacements/${arch}_n${n}_s0
        if [ ! -f "$rep/head.pt" ]; then
            log "train $arch n=$n"
            (cd "$VLA_ROOT/split" && python train_replacement.py --arch $arch --num_queries $n \
                --pairs_dir "$PAIRS" --out_root "$ROOT/replacements") || { log "FAILED train $arch n=$n"; continue; }
        fi
        res="$ROOT/evals/$SUITE/split_eval_seed7_10ep_from40_${arch}_n${n}.json"
        [ -f "$res" ] && { log "eval $arch n=$n done already"; continue; }
        log "closed loop $arch n=$n"
        (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite $SUITE --first_episode 40 \
            --episodes_per_task 10 --bundle "$rep" --tag ${arch}_n${n} --split_root "$ROOT/evals" 2>&1 \
            | grep -E '^Success|Traceback|Error') || log "FAILED eval $arch n=$n"
    done
done
log "ALL DONE"
