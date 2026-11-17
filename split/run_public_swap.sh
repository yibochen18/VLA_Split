#!/usr/bin/env bash
# Attack 1: the attacker lacks some parts of OFT and plugs in the public openvla-7b weights instead (no queries, no
# training). Weight comparison first, then closed loop on held-out starts 40-49 (real model on the same scenes:
# Spatial 95, Long 94; RESULTS.md section 10).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/public_swap
mkdir -p "$ROOT/evals"

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }

# Secret part -> what is swapped for public weights (tag:parts)
CONFIGS=(
    "llm09-16:llm:9-16"
    "llm01-08:llm:1-8"
    "llm01-16:llm:1-16"
    "vision:vision projector"
    "piece1:vision projector llm:1-8"
    "llm01-32:llm:1-32"
    "all:vision projector llm:1-32"
)

eval_run() {  # suite tag parts...
    local suite=$1 tag=$2; shift 2
    local out="$ROOT/evals/$suite/split_eval_seed7_10ep_from40_$tag.json"
    if [ -f "$out" ]; then log "$suite $tag done already"; return 0; fi
    wait_gpu
    log "$suite $tag: public $*"
    (cd "$VLA_ROOT/openvla-oft" && python ../split/run_split_eval.py --suite "$suite" --first_episode 40 \
        --episodes_per_task 10 --bundle "$VLA_SPLIT/$suite/model2" --tag "$tag" --split_root "$ROOT/evals" \
        --public_swap "$@" 2>&1 | grep -E '^Success|^Public|Traceback|Error')
}

log "weight comparison"
for suite in libero_spatial libero_10; do
    [ -f "$ROOT/diff_$suite.json" ] || (cd "$VLA_ROOT/openvla-oft" && \
        python ../split/public_diff.py --suite "$suite" --out "$ROOT/diff_$suite.json" > "$ROOT/diff_$suite.txt" 2>&1) \
        || log "diff $suite failed"
done

for suite in libero_spatial libero_10; do
    for c in "${CONFIGS[@]}"; do
        tag=${c%%:*}; parts=${c#*:}
        # shellcheck disable=SC2086
        eval_run "$suite" "$tag" $parts || log "FAILED $suite $tag"
    done
done
log "all done"
