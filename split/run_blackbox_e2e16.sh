#!/usr/bin/env bash
# Layers 1-16 end to end (RESULTS §17), plus a check of the recorded actions. `n100`: N = 100.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$VLA_DATA/blackbox
RECORD=$VLA_DATA/secret_lora/record
log() { echo "=== [$(date '+%F %T')] $* ==="; }
cd "$VLA_ROOT/openvla-oft" || exit 1


for suite in libero_10 libero_spatial; do
    out=$ROOT/record_check_$suite.json
    [ -f "$out" ] || { log "check recorded actions $suite"; python ../split/check_record_actions.py --suite $suite --n 1000 --out "$out" 2>&1 | grep -A9 '^{'; }
done

e2e() {  # suite kind n
    local suite=$1 init=L1-16_$2_n$3 t=L1-16_$2_n$3_e2e opt=adamw
    local out=$ROOT/fits/$suite/$t.pt res=$ROOT/evals/$suite/split_eval_seed7_10ep_from40_$t.json
    [ "$2" = tf4 ] && opt=adamw8bit
    if [ ! -f "$out" ]; then
        log "fit $suite $t"
        python ../split/fit_blackbox_e2e.py --suite "$suite" --init "$ROOT/fits/$suite/$init.pt" \
            --record_dir "$RECORD/$suite" --n_queries "$3" --optim "$opt" --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^start|^step 3000|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return; }
    fi
    [ -f "$res" ] && { log "eval $suite $t done already"; return; }
    log "eval $suite $t"
    python ../split/run_split_eval.py --suite "$suite" --first_episode 40 --episodes_per_task 10 \
        --bundle "$VLA_SPLIT/$suite/model2" --tag "$t" --split_root "$ROOT/evals" --blackbox "$out" 2>&1 \
        | grep -E '^Success|Traceback|Error' || log "FAILED eval $t"
}

if [ "$1" = n100 ]; then
    for suite in libero_10 libero_spatial; do for kind in tf2 tf4; do e2e $suite $kind 100; done; done
    log "all done"; exit
fi

e2e libero_10 tf4 5000
e2e libero_10 tf2 1000
e2e libero_10 tf4 1000
for n in 5000 1000; do for kind in tf2 tf4; do e2e libero_spatial $kind $n; done; done
log "all done"
