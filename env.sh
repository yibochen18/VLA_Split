# Machine-specific settings (data paths, conda env, GPU); a variable set before sourcing wins.

export VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLA_DATA="${VLA_DATA:-/mnt/ssd/yibo}"                  # weights, datasets, logs
export VLA_CONDA_DIR="${VLA_CONDA_DIR:-$VLA_DATA/miniconda3}"
export VLA_CONDA_ENV="${VLA_CONDA_ENV:-openvla-oft}"
export HF_HOME="${HF_HOME:-$VLA_DATA/hf}"                     # Hugging Face cache
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"      # pick a free GPU
export MUJOCO_GL="${MUJOCO_GL:-egl}"                          # headless rendering

# Output locations, all under VLA_DATA
export VLA_LOGS="$VLA_DATA/logs"
export VLA_ROLLOUTS="$VLA_DATA/rollouts"
export VLA_REFERENCE="$VLA_DATA/reference"
export VLA_SPLIT="$VLA_DATA/split"
export VLA_EXTRACTION="$VLA_DATA/extraction"

vla_activate() {
    source "$VLA_CONDA_DIR/etc/profile.d/conda.sh" && conda activate "$VLA_CONDA_ENV"
}
