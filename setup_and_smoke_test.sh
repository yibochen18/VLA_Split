#!/usr/bin/env bash
# One-time OpenVLA-OFT + LIBERO setup, then a short LIBERO-Spatial smoke test.
set -euo pipefail

source "$(dirname "$0")/env.sh"
SSD=$VLA_DATA
VLA=$VLA_ROOT
CONDA_DIR=$VLA_CONDA_DIR
ENV=$VLA_CONDA_ENV

mkdir -p "$HF_HOME" "$VLA_LOGS" "$VLA_ROLLOUTS"
step() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

step "1. HF_HOME in ~/.bashrc"
grep -q 'HF_HOME=' ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

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
# protobuf 3.20 for tf 2.15 (needs an older wandb)
pip install "tensorflow-metadata==1.15.0" "wandb==0.16.6"
# mujoco 3.17 fails robosuite 1.4.1's joint-type assertion when building LIBERO scenes
pip install "mujoco==3.3.0"
# 8-bit AdamW for the 4-layer black-box replacements (fit_blackbox*.py --optim adamw8bit)
pip install "bitsandbytes==0.43.3"
# LIBERO asks an interactive question on first import; answer N to use default paths
echo n | python -c "import libero.libero" || true

step "7. Sanity check"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

step "8. Smoke test: LIBERO-Spatial, 1 trial per task (10 episodes), GPU $CUDA_VISIBLE_DEVICES"
# Rollout videos are written to ./rollouts; keep them on the SSD
[ -e rollouts ] || ln -s "$VLA_ROLLOUTS" rollouts
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1 \
    --local_log_dir "$VLA_LOGS"

step "DONE"
