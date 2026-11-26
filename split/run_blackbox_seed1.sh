#!/usr/bin/env bash
# Second training seed at N = 5,000 for every black-box transformer cell (RESULTS §17).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$VLA_DATA/blackbox
RECORD=$VLA_DATA/secret_lora/record
RECORD9=$ROOT/record9
N=5000
log() { echo "=== [$(date '+%F %T')] $* ==="; }
cd "$VLA_ROOT/openvla-oft" || exit 1

evaluate() {  # suite tag
    local res=$ROOT/evals/$1/split_eval_seed7_10ep_from40_$2.json
    [ -f "$res" ] && { log "eval $1 $2 done already"; return; }
    log "eval $1 $2"
    python ../split/run_split_eval.py --suite "$1" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$1/model2" --tag "$2" --split_root "$ROOT/evals" --blackbox "$ROOT/fits/$1/$2.pt" 2>&1 \
        | grep -E '^Success|Traceback|Error' || log "FAILED eval $1 $2"
}

cell() {  # suite a b kind
    local suite=$1 a=$2 b=$3 kind=$4 rec=$RECORD
    [ "$a-$b" = 1-9 ] && rec=$RECORD9
    local t=L$a-${b}_${kind}_n${N}_seed1 out=$ROOT/fits/$suite/L$a-${b}_${kind}_n${N}_seed1.pt
    if [ ! -f "$out" ]; then
        log "fit $suite $t"
        python ../split/fit_blackbox.py --block "$a" "$b" --kind "$kind" --record_dir "$rec/$suite" --n_queries $N \
            --seed 1 --loss scaled --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^fitted|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return; }
    fi
    evaluate "$suite" "$t"
    [ "$a" = 1 ] || return
    local te=${t}_e2e oute=$ROOT/fits/$suite/${t}_e2e.pt opt=adamw
    [ "$kind" = tf4 ] && opt=adamw8bit
    if [ ! -f "$oute" ]; then
        log "fit $suite $te"
        python ../split/fit_blackbox_e2e.py --suite "$suite" --init "$out" --record_dir "$rec/$suite" --n_queries $N \
            --optim "$opt" --out "$oute" > "${oute%.pt}.log" 2>&1
        grep -E '^loaded|^start|^step 3000|Traceback|Error' "${oute%.pt}.log"
        [ -f "$oute" ] || { log "FAILED fit $te"; return; }
    fi
    evaluate "$suite" "$te"
}

for suite in libero_10 libero_spatial; do
    for blk in "1 16" "1 9" "9 16"; do
        for kind in tf2 tf4; do
            cell $suite $blk $kind
        done
    done
done
log "all done"
