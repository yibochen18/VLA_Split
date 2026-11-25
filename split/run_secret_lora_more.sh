#!/usr/bin/env bash
# Public weights + LoRA on layers 9-16 and 1-9, and public weights alone on 1-9 (RESULTS §17).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/secret_lora
PUB=$VLA_DATA/public_swap
RECORD9=$VLA_DATA/blackbox/record9
log() { echo "=== [$(date '+%F %T')] $* ==="; }
cd "$VLA_ROOT/openvla-oft" || exit 1

public_only() {  # suite
    local suite=$1 out="$PUB/evals/$1/split_eval_seed7_10ep_from40_llm01-09.json"
    [ -f "$out" ] && { log "$suite public llm01-09 done already"; return; }
    log "eval $suite public llm01-09"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag llm01-09 --split_root "$PUB/evals" --public_swap llm:1-9 2>&1 \
        | grep -E '^Success|^Public|Traceback|Error' || log "FAILED public $suite"
}

run() {  # suite tag record_dir n parts
    local suite=$1 tag=$2 rec=$3 n=$4 parts=$5
    local out=$ROOT/fits/$suite/${tag}_n$n res="$ROOT/evals/$suite/split_eval_seed7_10ep_from40_${tag}_n$n.json"
    if [ ! -f "$out/lora.pt" ]; then
        local total n_val; total=$(ls "$rec/$suite" | grep -c '\.pt$'); n_val=$(( total - n < 200 ? total - n : 200 ))
        mkdir -p "$out"; log "fit $suite $tag n=$n (val $n_val)"
        python ../split/fit_secret_lora.py --suite "$suite" --parts $parts --record_dir "$rec/$suite" \
            --n_queries "$n" --n_val "$n_val" --out "$out" > "$out.log" 2>&1
        grep -E '^public weights|^fitted|Traceback|Error' "$out.log"
        [ -f "$out/lora.pt" ] || { log "FAILED fit $suite $tag n=$n"; return; }
    fi
    [ -f "$res" ] && { log "eval $suite $tag n=$n done already"; return; }
    log "eval $suite $tag n=$n"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag "${tag}_n$n" --split_root "$ROOT/evals" \
        --public_swap $parts --public_lora "$out/lora.pt" 2>&1 | grep -E '^Success|^LoRA|Traceback|Error' \
        || log "FAILED eval $suite $tag n=$n"
}

for suite in libero_spatial libero_10; do
    public_only "$suite"
    for n in 100 1000 5000; do
        run "$suite" llm09-16 "$ROOT/record" "$n" "llm:9-16"
        run "$suite" llm01-09 "$RECORD9" "$n" "llm:1-9"
    done
done
log "all done"
