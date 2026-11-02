#!/usr/bin/env bash
# Full 500-episode LIBERO baselines for the remaining suites, one after another.
source "$(dirname "$0")/../env.sh" && vla_activate
cd "$VLA_ROOT/openvla-oft"
for suite in libero_object libero_goal libero_10; do
    ckpt=moojink/openvla-7b-oft-finetuned-${suite#libero_}
    [ $suite = libero_10 ] && ckpt=moojink/openvla-7b-oft-finetuned-libero-10
    [ $suite != libero_10 ] && ckpt=moojink/openvla-7b-oft-finetuned-libero-${suite#libero_}
    echo "=== [$(date '+%F %T')] $suite ($ckpt) ==="
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint $ckpt --task_suite_name $suite --seed 7 \
        --local_log_dir "$VLA_LOGS" 2>&1 | tee "$VLA_LOGS"/baseline_${suite}_seed7.log
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
