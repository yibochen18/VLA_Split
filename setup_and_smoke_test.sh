#!/usr/bin/env bash
# One-time setup for OpenVLA-OFT + LIBERO, then a short LIBERO-Spatial smoke test on GPU 1.
# Log: /mnt/ssd/yibo/logs/setup.log
set -euo pipefail

SSD=/mnt/ssd/yibo
VLA=/home/yibochen/VLA
CONDA_DIR=$SSD/miniconda3
ENV=openvla-oft

mkdir -p $SSD/hf $SSD/logs $SSD/rollouts
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

step "1. HF_HOME in ~/.bashrc"
grep -q 'HF_HOME=' ~/.bashrc || echo "export HF_HOME=$SSD/hf" >> ~/.bashrc
export HF_HOME=$SSD/hf

step "2. Miniconda at $CONDA_DIR"
if [ ! -x $CONDA_DIR/bin/conda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $SSD/miniconda.sh
    bash $SSD/miniconda.sh -b -p $CONDA_DIR
    rm $SSD/miniconda.sh
    $CONDA_DIR/bin/conda init bash
fi
source $CONDA_DIR/etc/profile.d/conda.sh
# Newer Miniconda requires accepting the default channels' ToS before non-interactive use
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

step "3. Conda env $ENV (python 3.10)"
conda env list | grep -q "^$ENV " || conda create -n $ENV python=3.10 -y
conda activate $ENV

step "4. PyTorch 2.2.0 + openvla-oft (pinned versions from LIBERO.md)"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
cd $VLA/openvla-oft
pip install -e .

step "5. flash-attn 2.5.5 (training only; eval has it commented out, so failure is non-fatal)"
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation || echo "WARNING: flash-attn install failed (not needed for eval)"

step "6. LIBERO + eval requirements"
# compat mode: the default (PEP 660) editable install leaves `libero` unimportable
pip install -e $VLA/LIBERO --config-settings editable_mode=compat
pip install -r experiments/robot/libero/libero_requirements.txt
# The requirements above pull in numpy 2 / opencv 5, which break torch 2.2 and tensorflow 2.15
pip install "numpy<2" "opencv-python==4.10.0.84"
# Newest tensorflow-metadata needs protobuf 5 (breaks tf 2.15); 1.15 pins protobuf 3.20, which needs an older wandb
pip install "tensorflow-metadata==1.15.0" "wandb==0.16.6"
# mujoco 3.17 fails robosuite 1.4.1's joint-type assertion when building LIBERO scenes
pip install "mujoco==3.3.0"
# LIBERO asks an interactive question on first import; answer N to use default paths
echo n | python -c "import libero.libero" || true

step "7. Sanity check"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

step "8. Smoke test: LIBERO-Spatial, 1 trial per task (10 episodes), GPU 1"
# Rollout videos are written to ./rollouts; keep them on the SSD
[ -e rollouts ] || ln -s $SSD/rollouts rollouts
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1 \
    --local_log_dir $SSD/logs

step "DONE"
