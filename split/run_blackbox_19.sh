#!/usr/bin/env bash
# Layers 1-9 black box (RESULTS §17): 2-/4-layer transformers at N = 100/1,000/5,000, simple then end to end.
source "$(dirname "$0")/../env.sh" && vla_activate
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$VLA_DATA/blackbox
RECORD=$VLA_DATA/secret_lora/record
RECORD9=$ROOT/record9
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

simple() {  # suite kind n
    local suite=$1 t=L1-9_$2_n$3 out=$ROOT/fits/$1/L1-9_$2_n$3.pt
    if [ ! -f "$out" ]; then
        log "fit $suite $t"
        python ../split/fit_blackbox.py --block 1 9 --kind "$2" --record_dir "$RECORD9/$suite" --n_queries "$3" \
            --loss scaled --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^fitted|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return 1; }
    fi
    evaluate "$suite" "$t"
}

e2e() {  # suite init-tag n
    local suite=$1 init=$2 n=$3 t=$2_e2e out=$ROOT/fits/$1/$2_e2e.pt opt=adamw
    [[ $init == *tf4* ]] && opt=adamw8bit
    if [ ! -f "$out" ]; then
        log "fit $suite $t"
        python ../split/fit_blackbox_e2e.py --suite "$suite" --init "$ROOT/fits/$suite/$init.pt" \
            --record_dir "$RECORD9/$suite" --n_queries "$n" --optim "$opt" --out "$out" > "${out%.pt}.log" 2>&1
        grep -E '^loaded|^start|^step 3000|Traceback|Error' "${out%.pt}.log"
        [ -f "$out" ] || { log "FAILED fit $t"; return; }
    fi
    evaluate "$suite" "$t"
}

record9() {  # suite: re-record at layer 9
    local suite=$1 dir=$RECORD9/$1
    [ -f "$dir/.done" ] && { log "$suite layer 9 recorded already"; return 0; }
    rm -rf "$dir"; log "record layer 9 $suite"
    python ../split/run_split_eval.py --suite "$suite" --episodes_per_task 20 \
        --bundle "$VLA_SPLIT/$suite/model2" --split_root "$ROOT/record9_evals" --record_full "$dir" \
        --record_full_layers 9 2>&1 | grep -E '^Success|Traceback|Error' && touch "$dir/.done"
    log "$suite: $(ls "$dir" | grep -c '\.pt$') queries (attack-2 recording: $(ls "$RECORD/$suite" | grep -c '\.pt$'))"
    [ -f "$dir/.done" ]
}

S=libero_spatial
e2e $S L1-9_tf4_n5000_seed1 5000
e2e $S L1-9_tf4_n5000 5000

ok=0
for t in L1-9_tf2_n5000_seed1 L1-9_tf4_n5000_seed1 L1-9_tf4_n5000 L1-9_tf2_n1000; do
    f=$ROOT/evals/$S/split_eval_seed7_10ep_from40_${t}_e2e.json
    s=$(python -c "import json;print(round(100*json.load(open('$f'))['success_rate']))" 2>/dev/null || echo 0)
    log "gate: ${t}_e2e = $s"
    [ "$s" -ge 85 ] && ok=$((ok + 1))
done
[ $ok -ge 3 ] || { log "gate not passed ($ok/4 >= 85); stopping"; exit 0; }
log "gate passed ($ok/4)"

for kind in tf2 tf4; do simple $S $kind 100 && e2e $S L1-9_${kind}_n100 100; done
e2e $S L1-9_tf4_n1000 1000
L=libero_10
record9 $L || { log "RECORD FAILED"; exit 1; }
for n in 1000 5000 100; do
    for kind in tf2 tf4; do simple $L $kind $n && e2e $L L1-9_${kind}_n$n $n; done
done
log "all done"
