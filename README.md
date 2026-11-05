# VLA Split

Splitting OpenVLA-OFT into two models that run in sequence: Model 1 (vision encoders + Llama 2 backbone) produces an intermediate representation, and Model 2 (the MLP action head) turns it into robot actions. Evaluated on the LIBERO simulation benchmark. The linear-replacement study is repeated on π0.5, whose action module is a flow-matching transformer rather than an MLP head.

Main findings: the split is exact; Model 2 can be copied from ~100 observed queries; one linear map fitted from observed traffic replaces Model 2 with no loss in success or robustness, and from layer 24 also replaces the top 8 Llama layers (~15% faster per query). On π0.5, one linear map replaces the 300M-parameter action expert with no significant loss overall (96.0% vs. 96.7%) and 73% faster queries.

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: baseline, cut A split, extraction tests, linear-head replacement, depth sweep, compute and observation cost, robustness, CPU TEE budget, π0.5 |

## Layout

```
env.sh                    machine-specific settings: data location, conda env, GPU (single source of truth)
split/                    our code (openvla-oft and openpi are used unmodified)
setup_and_smoke_test.sh   one-time environment setup + 10-episode smoke test
openvla-oft/, LIBERO/     external repos, not tracked (see below)
openpi/                   π0.5 (Physical Intelligence), not tracked; runs in its own environment
```

Model weights, logs, rollout videos and datasets live outside the repo under `$VLA_DATA` (`/mnt/ssd/yibo` on our server).

## Configuration

Every script reads its locations, conda env and GPU from [env.sh](env.sh) (Python via `split/paths.py`); nothing else hardcodes a path. On another machine or account, edit the defaults there, or override any of them per run:

```bash
VLA_DATA=/data/me CUDA_VISIBLE_DEVICES=0 bash split/run_baselines.sh
```

Interactive use: `source env.sh && vla_activate`, then run scripts from `openvla-oft/` as their docstrings show.

## External dependencies

Clone these into the repo root at the pinned commits, check `env.sh`, then run setup:

```bash
git clone https://github.com/moojink/openvla-oft.git && git -C openvla-oft checkout e4287e94541f459edc4feabc4e181f537cd569a8
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
bash setup_and_smoke_test.sh
```

`setup_and_smoke_test.sh` documents the package pins needed on top of the upstream instructions.

For π0.5, openpi runs in its own uv environment (Python 3.11) under `$VLA_OPENPI`; `vla_openpi <cmd>` (from `env.sh`) runs a command in it:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git && git -C openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
pip install --user uv                                   # if missing
source env.sh && cd openpi
export UV_CACHE_DIR=$VLA_OPENPI/uv-cache UV_PROJECT_ENVIRONMENT=$VLA_OPENPI/venv UV_PYTHON_INSTALL_DIR=$VLA_OPENPI/python
GIT_LFS_SKIP_SMUDGE=1 uv sync
VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --reinstall --link-mode=copy transformers==4.53.2   # copies, so the patch below stays in this env
cp -r src/openpi/models_pytorch/transformers_replace/* $UV_PROJECT_ENVIRONMENT/lib/python3.11/site-packages/transformers/
vla_openpi python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"
vla_openpi examples/convert_jax_model_to_pytorch.py --checkpoint_dir $VLA_OPENPI/data/openpi-assets/checkpoints/pi05_libero \
    --config_name pi05_libero --output_path $VLA_OPENPI/checkpoints/pi05_libero_pytorch
cp -r $VLA_OPENPI/data/openpi-assets/checkpoints/pi05_libero/assets $VLA_OPENPI/checkpoints/pi05_libero_pytorch/
```

## Experiments

Each script writes only under `$VLA_DATA`; run them in this order (RESULTS.md section in brackets):

```
split/run_baselines.sh, split/run_v3_all.sh      baselines and the bitwise split check (§1-5)
split/run_extraction.sh, split/run_layers.sh     copying Model 2 at cut A and cuts B-k (§6-7)
split/run_linear_record.sh, run_linear_cv.sh     linear head, 5-fold cross-validation (§8)
split/run_linear_layers.sh, run_linear_refs.sh   depth sweep and same-scene references (§9-10)
split/run_linear_compute.sh                      compute saved and observation needed (§11)
split/run_linear_followup.sh                     limits of depth and controls (§12)
split/run_robustness.sh                          robustness under distribution shift (§13)
split/bench_tee.py                               what fits in a CPU TEE (§14)
split/pi05_check_state.py, run_pi05_baselines.sh π0.5: checks and baselines with recording (§15)
split/ridge_pi05.py                              π0.5: choosing the linear map's input (§15)
split/run_pi05_cv.sh, pi05_cv_summary.py         π0.5: 5-fold closed loop vs. same-scene π0.5 (§15)
```
