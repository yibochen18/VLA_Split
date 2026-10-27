#!/usr/bin/env bash
# Full 500-episode LIBERO baselines for the remaining suites, one after another on GPU 1.
source /mnt/ssd/yibo/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
cd /home/yibochen/VLA/openvla-oft
export HF_HOME=/mnt/ssd/yibo/hf CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl
for suite in libero_object libero_goal libero_10; do
    ckpt=moojink/openvla-7b-oft-finetuned-${suite#libero_}
    [ $suite = libero_10 ] && ckpt=moojink/openvla-7b-oft-finetuned-libero-10
    [ $suite != libero_10 ] && ckpt=moojink/openvla-7b-oft-finetuned-libero-${suite#libero_}
    echo "=== [$(date '+%F %T')] $suite ($ckpt) ==="
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint $ckpt --task_suite_name $suite --seed 7 \
        --local_log_dir /mnt/ssd/yibo/logs 2>&1 | tee /mnt/ssd/yibo/logs/baseline_${suite}_seed7.log
done
echo "=== [$(date '+%F %T')] ALL DONE ==="
