#!/usr/bin/env bash
# Black-box follow-up (RESULTS §17): longer training, all queries, and layers 1-9.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/blackbox
RECORD=$VLA_DATA/secret_lora/record
RECORD9=$VLA_DATA/blackbox/record9
declare -A RECORD_EPISODES=([libero_spatial]=40 [libero_10]=20)

log() { echo "=== [$(date '+%F %T')] $* ==="; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 7500 ]; do sleep 60; done; }
cd "$VLA_ROOT/openvla-oft" || exit 1

record9() {  # suite: re-record at layer 9
    local suite=$1 dir=$RECORD9/$1
    if [ -f "$dir/.done" ]; then log "$suite layer 9 recorded already"; return 0; fi
    rm -rf "$dir"; wait_gpu
    log "record layer 9 $suite"
    python ../split/run_split_eval.py --suite "$suite" --episodes_per_task "${RECORD_EPISODES[$suite]}" \
        --bundle "$VLA_SPLIT/$suite/model2" --split_root "$ROOT/record9_evals" --record_full "$dir" \
        --record_full_layers 9 2>&1 | grep -E '^Success|Traceback|Error' && touch "$dir/.done"
    log "$suite: $(ls "$dir" | grep -c '\.pt$') queries (attack-2 recording: $(ls "$RECORD/$suite" | grep -c '\.pt$'))"
    python - "$dir" "$RECORD/$suite" <<'EOF'
import random, sys, torch
from pathlib import Path
new, old = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name for p in new.glob("*.pt"))
same = names == sorted(p.name for p in old.glob("*.pt"))
diff = max(float((torch.load(new / n)["inputs_embeds"].float() - torch.load(old / n)["inputs_embeds"].float()).abs().max())
           for n in random.Random(0).sample(names, 50)) if same else None
print(f"same query files: {same}; max |inputs_embeds diff| on 50: {diff}")
EOF
    [ -f "$dir/.done" ]
}

fit() {  # suite a b kind n steps record_dir
    local suite=$1 a=$2 b=$3 kind=$4 n=$5 steps=$6 rec=$7 tag
    tag=$(tag "$@")
    local out=$ROOT/fits/$suite/$tag.pt
    if [ -f "$out" ]; then log "fit $suite $tag done already"; return 0; fi
    mkdir -p "$(dirname "$out")"
    wait_gpu
    log "fit $suite $tag"
    local opt=adamw; [ "$kind" = tf8 ] && opt=adamw8bit
    python ../split/fit_blackbox.py --block "$a" "$b" --kind "$kind" --record_dir "$rec" --n_queries "$n" \
        --steps "$steps" --optim "$opt" --out "$out" > "${out%.pt}.log" 2>&1
    grep -E '^loaded|^identity|^fitted|Traceback|Error' "${out%.pt}.log"
    [ -f "$out" ]
}

tag() {  # names as run_blackbox.sh
    local t=L$2-$3_$4_n$5; [ "$6" != 10000 ] && t=${t}_s$(($6 / 1000))k; echo "$t"
}

eval_run() {
    local suite=$1 tag; tag=$(tag "$@")
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
    nmax=$(( $(ls "$RECORD/$suite" | grep -c '\.pt$') - 200 ))
    for kind in tf8 tf2; do
        for blk in "9 16" "1 16"; do
            # shellcheck disable=SC2086
            run "$suite" $blk "$kind" "$nmax" 30000 "$RECORD/$suite"
        done
    done
    record9 "$suite" || { log "RECORD FAILED $suite"; continue; }
    run "$suite" 1 9 identity 0 10000 "$RECORD9/$suite"
    for n in 1000 5000; do for kind in tf2 tf4; do run "$suite" 1 9 "$kind" "$n" 10000 "$RECORD9/$suite"; done; done
    for kind in tf2 tf8; do run "$suite" 1 9 "$kind" "$nmax" 30000 "$RECORD9/$suite"; done
done
log "all done"
