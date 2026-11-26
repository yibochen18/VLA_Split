#!/usr/bin/env bash
# First end-to-end fine-tunes of black-box replacements (RESULTS §17).
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
ROOT=$VLA_DATA/blackbox
log() { echo "=== [$(date '+%F %T')] $* ==="; }
cd "$VLA_ROOT/openvla-oft" || exit 1

run() {  # suite init-tag n record_dir
    local suite=$1 init=$2 n=$3 rec=$4 t=$2_e2e
    local fits=$ROOT/fits/$suite out=$ROOT/fits/$suite/$t.pt
    if [ ! -f "$out" ]; then
        log "fit $suite $t"
        python ../split/fit_blackbox_e2e.py --suite "$suite" --init "$fits/$init.pt" --record_dir "$rec" \
            --n_queries "$n" --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^start|^step 3000|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return; }
    fi
    python ../split/diagnose_blackbox.py --suite "$suite" --n 200 --fits "$init" "$t" \
        --out "$ROOT/diagnose_$t.json" 2>&1 | grep -E '^L'
    log "eval $suite $t"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag "$t" --split_root "$ROOT/evals" --blackbox "$out" 2>&1 \
        | grep -E '^Success|Traceback|Error' || log "FAILED eval $t"
}

S=libero_spatial
run $S L1-9_tf2_n5000_seed1 5000 "$ROOT/record9/$S"
run $S L1-9_tf2_n1000 1000 "$ROOT/record9/$S"
run libero_10 L1-16_tf2_n5000 5000 "$VLA_DATA/secret_lora/record/libero_10"
log "all done"
