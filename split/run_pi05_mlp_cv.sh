#!/usr/bin/env bash
# π0.5: 5-fold CV of a small MLP (mlp_pi05.py) in place of the action expert, as run_pi05_cv.sh does for the linear
# map; reuses its same-scene π0.5 references. Finished fits and folds are skipped, so a rerun resumes.
#   bash split/run_pi05_mlp_cv.sh        MLP_OUT=<dir> MLP_ARGS="--norm layernorm" bash split/run_pi05_mlp_cv.sh
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_DATA/pi05
OUT=${MLP_OUT:-$ROOT/mlp_cv_std}
MLP_ARGS=${MLP_ARGS:-}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
WORKERS=${WORKERS:-2}  # each π0.5 server takes ~8 GB
mkdir -p "$OUT"
log() { echo "=== [$(date '+%F %T')] $* ==="; }

for suite in $SUITES; do
    for f in 0 1 2 3 4; do
        v=$(( (f + 1) % 5 ))
        head="$OUT/heads/${suite}_fold$f"
        [ -f "$head/config.json" ] && continue
        log "fit $suite fold $f"
        (cd "$VLA_ROOT/split" && python mlp_pi05.py --record_dir "$ROOT/record/$suite" --cache "$ROOT/features/pool4/$suite.pt" \
            --test_from $((10 * f)) --test_to $((10 * f + 10)) --val_from $((10 * v)) --val_to $((10 * v + 10)) \
            --out "$head" $MLP_ARGS 2>&1 | grep -E '^mlp|steps|Traceback|Error')
    done
done

eval_fold() {  # suite fold
    local suite=$1 f=$2 from=""
    [ "$2" != 0 ] && from="_from$((10 * $2))"
    local result="$OUT/evals/$suite/pi05_eval_seed7_10ep${from}_mlp_fold$f.json"
    [ -f "$result" ] && return
    log "mlp closed loop $suite fold $f"
    cd "$VLA_ROOT/openvla-oft" && python ../split/run_pi05_eval.py --suite "$suite" --first_episode $((10 * f)) \
        --episodes_per_task 10 --mlp "$OUT/heads/${suite}_fold$f" --tag "mlp_fold$f" --out_root "$OUT/evals" \
        --socket "/tmp/vla_pi05_mlp_${suite}_$f.sock" --save_actions 2>&1 | grep -E '^Success|^Latency|Traceback|Error' \
        | sed "s/^/$suite fold $f: /"
}
export -f eval_fold log
export OUT VLA_ROOT

for suite in $SUITES; do for f in 0 1 2 3 4; do echo "$suite $f"; done; done | xargs -P "$WORKERS" -L 1 bash -c 'eval_fold $0 $1'
log "ALL DONE"
python "$VLA_ROOT/split/pi05_cv_summary.py" mlp "$OUT"
