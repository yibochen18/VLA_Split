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
export VLA_OPENPI="$VLA_DATA/openpi"                            # pi0.5: openpi env, checkpoints, caches

vla_activate() {
    source "$VLA_CONDA_DIR/etc/profile.d/conda.sh" && conda activate "$VLA_CONDA_ENV"
}

# Run a command in openpi's own environment (Python 3.11, uv); openpi itself is unmodified apart from its transformers patch
vla_openpi() {
    UV_CACHE_DIR="$VLA_OPENPI/uv-cache" UV_PROJECT_ENVIRONMENT="$VLA_OPENPI/venv" UV_PYTHON_INSTALL_DIR="$VLA_OPENPI/python" \
        OPENPI_DATA_HOME="$VLA_OPENPI/data" "${VLA_UV:-$HOME/.local/bin/uv}" run --no-sync --project "$VLA_ROOT/openpi" "$@"
}
