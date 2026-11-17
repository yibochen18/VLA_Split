#!/usr/bin/env bash
# Black-box attack: the attacker has no related weights and trains a replacement for a secret block of Llama
# layers from scratch on recorded (input, output) traffic (blackbox.py, fit_blackbox.py). Uses the recordings of
# run_secret_lora.sh. Blocks 9-16 and 1-16; identity (block dropped), linear, MLP, 2- and 4-layer transformers;
# N = 100, 1,000, 5,000; closed loop on held-out starts 40-49 (real model: Spatial 95, Long 94).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/blackbox
RECORD=$VLA_DATA/secret_lora/record
NS="100 1000 5000"
BLOCKS=("9 16" "1 16")
KINDS="linear mlp tf2 tf4"

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
cd "$VLA_ROOT/openvla-oft" || exit 1

fit() {  # suite a b kind n
    local suite=$1 a=$2 b=$3 kind=$4 n=$5
    local out=$ROOT/fits/$suite/L${a}-${b}_${kind}_n$n.pt
    if [ -f "$out" ]; then log "fit $suite $a-$b $kind n=$n done already"; return 0; fi
    mkdir -p "$(dirname "$out")"
    wait_gpu
    log "fit $suite $a-$b $kind n=$n"
    python ../split/fit_blackbox.py --block "$a" "$b" --kind "$kind" --record_dir "$RECORD/$suite" \
        --n_queries "$n" --out "$out" > "${out%.pt}.log" 2>&1
    grep -E '^loaded|^identity|^fitted|Traceback|Error' "${out%.pt}.log"
    [ -f "$out" ]
}

eval_run() {  # suite a b kind n
    local suite=$1 a=$2 b=$3 kind=$4 n=$5 tag=L$2-$3_$4_n$5
    local res="$ROOT/evals/$suite/split_eval_seed7_10ep_from40_$tag.json"
    if [ -f "$res" ]; then log "eval $suite $tag done already"; return 0; fi
    wait_gpu
    log "eval $suite $tag"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag "$tag" --split_root "$ROOT/evals" \
        --blackbox "$ROOT/fits/$suite/$tag.pt" 2>&1 | grep -E '^Success|^Black|Traceback|Error'
}

run() { fit "$@" && eval_run "$@" || log "FAILED $*"; }

for suite in libero_spatial libero_10; do
    [ -f "$RECORD/$suite/.done" ] || { log "no recording for $suite"; continue; }
    for blk in "${BLOCKS[@]}"; do
        # shellcheck disable=SC2086
        run "$suite" $blk identity 0
    done
    for n in $NS; do
        for blk in "${BLOCKS[@]}"; do
            for kind in $KINDS; do
                # shellcheck disable=SC2086
                run "$suite" $blk "$kind" "$n"
            done
        done
    done
done
log "all done"
