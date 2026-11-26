#!/usr/bin/env bash
# Full 500-episode LIBERO baselines: Object, Goal, Long (Spatial: README, "Running on LIBERO" step 1).
source "$(dirname "$0")/../env.sh" && vla_activate
cd "$VLA_ROOT/openvla-oft"
for suite in libero_object libero_goal libero_10; do
    ckpt=moojink/openvla-7b-oft-finetuned-${suite/_/-}
    echo "=== [$(date '+%F %T')] $suite ($ckpt) ==="
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint $ckpt --task_suite_name $suite --seed 7 \
        --local_log_dir "$VLA_LOGS" 2>&1 | tee "$VLA_LOGS"/baseline_${suite}_seed7.log
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
