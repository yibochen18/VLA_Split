#!/usr/bin/env bash
# π0.5: 5-fold cross-validation of the linear map in place of the action expert (as run_linear_cv.sh), all suites,
# in parallel with the same-scene π0.5 references (folds 1-4; fold 0 is the baseline run, which also starts at 0).
source "$(dirname "$0")/../env.sh" && vla_activate
ROOT=$VLA_DATA/pi05
cd "$VLA_ROOT/openvla-oft"
log() { echo "=== [$(date '+%F %T')] $* ==="; }

linear_folds() {
    for suite in libero_spatial libero_object libero_goal libero_10; do
        for f in 4 0 1 2 3; do
            v=$(( (f + 1) % 5 ))
            head="$ROOT/cv/heads/${suite}_fold$f"
            log "fit $suite fold $f"
            python ../split/ridge_pi05.py --record_dir "$ROOT/record/$suite" --layer layer18_in --pool 4 \
                --test_from $((10 * f)) --test_to $((10 * f + 10)) --val_from $((10 * v)) --val_to $((10 * v + 10)) \
                --out "$head" 2>&1 | grep -E '^layer18_in|Traceback|Error'
            log "linear closed loop $suite fold $f"
            python ../split/run_pi05_eval.py --suite $suite --first_episode $((10 * f)) --episodes_per_task 10 \
                --linear "$head" --tag linear_fold$f --out_root "$ROOT/cv/evals" --socket /tmp/vla_pi05_cv_linear.sock \
                --save_actions 2>&1 | grep -E '^Success|^Latency|Traceback|Error'
        done
    done
    log "LINEAR DONE"
}

references() {
    for suite in libero_spatial libero_object libero_goal libero_10; do
        for f in 1 2 3 4; do
            log "pi05 reference $suite fold $f"
            python ../split/run_pi05_eval.py --suite $suite --first_episode $((10 * f)) --episodes_per_task 10 \
                --tag ref_fold$f --out_root "$ROOT/cv/evals" --socket /tmp/vla_pi05_cv_ref.sock \
                --save_actions 2>&1 | grep -E '^Success|^Latency|Traceback|Error'
        done
    done
    log "REFERENCES DONE"
}

linear_folds > "$VLA_LOGS/pi05_cv_linear.log" 2>&1 &
references > "$VLA_LOGS/pi05_cv_refs.log" 2>&1 &
wait
log "ALL DONE"
