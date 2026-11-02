#!/usr/bin/env bash
# V3 on every suite (unsplit reference, then split replay); also times the unsplit model.
source "$(dirname "$0")/../env.sh" && vla_activate
cd "$VLA_ROOT/openvla-oft"
declare -A CKPT=(
    [libero_spatial]=moojink/openvla-7b-oft-finetuned-libero-spatial
    [libero_object]=moojink/openvla-7b-oft-finetuned-libero-object
    [libero_goal]=moojink/openvla-7b-oft-finetuned-libero-goal
    [libero_10]=moojink/openvla-7b-oft-finetuned-libero-10
)
echo "=== [$(date '+%F %T')] unsplit end-to-end latency (libero_spatial) ==="
python ../split/probe_baseline.py --suite libero_spatial --checkpoint ${CKPT[libero_spatial]} --passes 1 --no_component_timers
for suite in libero_object libero_goal libero_10; do
    echo "=== [$(date '+%F %T')] reference: $suite ==="
    python ../split/probe_baseline.py --suite $suite --checkpoint ${CKPT[$suite]} --passes 1
    echo "=== [$(date '+%F %T')] V3: $suite ==="
    python ../split/run_split_eval.py --suite $suite --episodes_per_task 1 \
        --reference "$VLA_REFERENCE"/${suite}_seed7_1ep.pt
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
