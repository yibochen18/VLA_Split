#!/usr/bin/env bash
# Attack 2: the attacker records the secret part's input and output (all tokens) and fits a rank-32 LoRA on the
# public openvla-7b weights to reproduce it, then plugs it in. Secret parts that broke under attack 1
# (run_public_swap.sh); N = 100, 1,000, 5,000 observed queries; closed loop on held-out starts 40-49
# (real model on the same scenes: Spatial 95, Long 94).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/secret_lora
NS="100 1000 5000"
CONFIGS=(
    "llm01-16:llm:1-16"
    "vision:vision projector"
    "piece1:vision projector llm:1-8"
)
declare -A RECORD_EPISODES=([libero_spatial]=40 [libero_10]=20)   # starts 0..n-1, all 10 tasks

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
cd "$VLA_ROOT/openvla-oft" || exit 1

record() {  # suite
    local suite=$1 dir=$ROOT/record/$1
    if [ -f "$dir/.done" ]; then log "$suite recorded already"; return 0; fi
    rm -rf "$dir"; wait_gpu
    log "record $suite, starts 0-$((RECORD_EPISODES[$suite] - 1))"
    python ../split/run_split_eval.py --suite "$suite" --episodes_per_task "${RECORD_EPISODES[$suite]}" \
        --bundle "$VLA_SPLIT/$suite/model2" --split_root "$ROOT/record_evals" --record_full "$dir" \
        --record_full_layers 8 16 2>&1 | grep -E '^Success|Traceback|Error' && touch "$dir/.done"
    log "$suite: $(ls "$dir" | wc -l) queries, $(du -sh "$dir" | cut -f1)"
}

fit() {  # suite tag n parts...
    local suite=$1 tag=$2 n=$3; shift 3
    local out=$ROOT/fits/$suite/${tag}_n$n
    if [ -f "$out/lora.pt" ]; then log "fit $suite $tag n=$n done already"; return 0; fi
    local total; total=$(ls "$ROOT/record/$suite" | grep -c '\.pt$')
    local n_val=$(( total - n < 200 ? total - n : 200 ))
    mkdir -p "$out"
    wait_gpu
    log "fit $suite $tag n=$n (val $n_val)"
    python ../split/fit_secret_lora.py --suite "$suite" --parts "$@" --record_dir "$ROOT/record/$suite" \
        --n_queries "$n" --n_val "$n_val" --out "$out" > "$out.log" 2>&1
    grep -E '^public weights|^fitted|Traceback|Error' "$out.log"
    [ -f "$out/lora.pt" ]
}

eval_run() {  # suite tag n parts...
    local suite=$1 tag=$2 n=$3; shift 3
    local res="$ROOT/evals/$suite/split_eval_seed7_10ep_from40_${tag}_n$n.json"
    if [ -f "$res" ]; then log "eval $suite $tag n=$n done already"; return 0; fi
    wait_gpu
    log "eval $suite $tag n=$n"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag "${tag}_n$n" --split_root "$ROOT/evals" \
        --public_swap "$@" --public_lora "$ROOT/fits/$suite/${tag}_n$n/lora.pt" 2>&1 | grep -E '^Success|^LoRA|Traceback|Error'
}

for suite in libero_spatial libero_10; do
    record "$suite" || { log "RECORD FAILED $suite"; continue; }
    for n in $NS; do
        for c in "${CONFIGS[@]}"; do
            tag=${c%%:*}; parts=${c#*:}
            # shellcheck disable=SC2086
            fit "$suite" "$tag" "$n" $parts && eval_run "$suite" "$tag" "$n" $parts || log "FAILED $suite $tag n=$n"
        done
    done
done
log "all done"
