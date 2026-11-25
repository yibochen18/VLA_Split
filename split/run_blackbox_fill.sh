#!/usr/bin/env bash
# Black-box gaps (RESULTS §17): extra seeds, tf8 / 30k-step runs, per-token maps for layers 1-9.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/blackbox
FITS=$ROOT/fits
EVALS=$ROOT/evals
SUITE=libero_spatial
RECORD=$VLA_DATA/secret_lora/record
RECORD9=$ROOT/record9

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
cd "$VLA_ROOT/openvla-oft" || exit 1

tag() {  # a b kind n steps seed
    local t=L$1-$2_$3_n$4
    [ "$5" != 10000 ] && t=${t}_s$(($5 / 1000))k
    [ "$6" != 0 ] && t=${t}_seed$6
    echo "$t"
}

run() {  # a b kind n steps seed
    local a=$1 b=$2 kind=$3 n=$4 steps=$5 seed=$6 t rec=$RECORD
    t=$(tag "$@"); [ "$a-$b" = 1-9 ] && rec=$RECORD9
    local out=$FITS/$SUITE/$t.pt res=$EVALS/$SUITE/split_eval_seed7_10ep_from40_$t.json
    if [ ! -f "$out" ]; then
        wait_gpu; log "fit $t"
        local opt=adamw; [ "$kind" = tf8 ] && opt=adamw8bit
        python ../split/fit_blackbox.py --block "$a" "$b" --kind "$kind" --record_dir "$rec/$SUITE" --n_queries "$n" \
            --steps "$steps" --seed "$seed" --optim "$opt" --loss scaled --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^fitted|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return; }
    fi
    [ -f "$res" ] && { log "$t done already"; return; }
    wait_gpu; log "eval $t"
    python ../split/run_split_eval.py --suite "$SUITE" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$SUITE/model2" --tag "$t" --split_root "$EVALS" --blackbox "$out" 2>&1 \
        | grep -E '^Success|Traceback|Error' || log "FAILED eval $t"
}

nmax=$(( $(ls "$RECORD/$SUITE" | grep -c '\.pt$') - 200 ))
for seed in 1 2; do
    run 1 9 tf2 5000 10000 $seed
    run 1 9 tf4 5000 10000 $seed
    run 1 16 tf2 5000 10000 $seed
done
run 1 9 tf8 "$nmax" 30000 0
for blk in "1 9" "1 16" "9 16"; do
    # shellcheck disable=SC2086
    run $blk tf4 "$nmax" 30000 0
    for n in 1000 5000; do
        # shellcheck disable=SC2086
        run $blk tf8 $n 10000 0
    done
done
for kind in linear mlp; do
    for n in 100 1000 5000; do run 1 9 $kind $n 10000 0; done
done
log "all done"
