# VLA Split

Does keeping part of a VLA secret (in a TEE or on a server) protect it from the robot's owner? We split OpenVLA-OFT into
Model 1 (vision + Llama 2) and Model 2 (action head), hide one part at a time, and test whether an attacker who sees
everything outside it can rebuild a working replacement. Evaluated in closed loop on LIBERO. Side studies: π0.5 and
LIBERO-Plus. Full numbers: [docs/RESULTS.md](docs/RESULTS.md); interactive results: [vlasplit.netlify.app](https://vlasplit.netlify.app/).

## Findings

- **No secret part protects OFT.** An attacker rebuilt every part we hid (real model: Spatial 95, Long 94).
- **Action head:** copied from ~100 observed queries; one closed-form linear map replaces it with no loss.
  Fitted on layer 24, the same map also replaces the top 8 Llama layers (~15% faster per query).
- **Beginning/middle/first half (layers 1–9, 9–16, 1–16), vision encoder:** public `openvla-7b` weights plus a LoRA
  fitted on 100–1,000 queries come within ~3 points. With no related weights, a small transformer trained from scratch
  on 5,000 queries gets within ~5 points, except layers 1–16 on Long (79–89). Blocks starting at layer 1 need
  end-to-end training through the attacker's own later layers.
- **Deployment:** only a head-sized part fits an edge CPU TEE, and it is the easiest to copy.
- **π0.5:** a linear map replaces the action expert on LIBERO (96.0% vs. 96.7%) but loses 14.8 pp off-distribution on
  LIBERO-Plus, where OFT's linear head loses only 1.1 pp.

## Setup

All paths, the conda env and the GPU come from [env.sh](env.sh) (defaults are for our server: edit them, or override
per run, e.g. `VLA_DATA=/data/me CUDA_VISIBLE_DEVICES=0 bash split/run_baselines.sh`). Weights, logs and results live
under `$VLA_DATA`, outside the repo. Our code is in `split/`; upstream repos are used unmodified.

```bash
git clone https://github.com/moojink/openvla-oft.git && git -C openvla-oft checkout e4287e94541f459edc4feabc4e181f537cd569a8
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
bash setup_and_smoke_test.sh            # env + package pins + 10-episode smoke test
source env.sh && vla_activate           # then run scripts from openvla-oft/
```

<details><summary>π0.5 (openpi, own uv env; <code>vla_openpi &lt;cmd&gt;</code> runs in it)</summary>

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
</details>

<details><summary>LIBERO-Plus (clone of the OFT env; <code>vla_activate_plus</code>; don't install its requirements.txt)</summary>

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git && git -C LIBERO-plus checkout 4976dc3
source env.sh
conda create -n $VLA_CONDA_ENV_PLUS --clone $VLA_CONDA_ENV && conda activate $VLA_CONDA_ENV_PLUS
pip uninstall -y libero; pip install --no-deps -e LIBERO-plus
pip install wand scikit-image && conda install -c conda-forge imagemagick    # sensor-noise (motion blur) variants
# assets (9.5 GB): download assets.zip from huggingface.co/datasets/Sylvest/LIBERO-plus and unzip to $VLA_DATA/libero_plus/assets
ln -s $VLA_DATA/libero_plus/assets LIBERO-plus/libero/libero/assets
mkdir -p $VLA_DATA/libero_plus/config && B=$PWD/LIBERO-plus/libero/libero && cat > $VLA_DATA/libero_plus/config/config.yaml <<EOF
assets: $B/./assets
bddl_files: $B/./bddl_files
benchmark_root: $B
datasets: $B/../datasets
init_states: $B/./init_files
EOF
```
</details>

## Reproducing

Run in this order; each writes under `$VLA_DATA` (RESULTS.md section in brackets).

```
# Baselines and split
split/export_model2.py --suite <all four>          Model 2 bundles, needed by everything below
split/run_baselines.sh, run_v3_all.sh              baselines, bitwise split check (§1-5; RESULTS "Reproducing")
split/run_split_eval.py --suite libero_spatial \
    --save_pairs $VLA_EXTRACTION/libero_spatial/pairs   Spatial split run recording cut-A pairs (§5-6)

# Action head and top layers
split/run_extraction.sh, run_layers.sh             copying Model 2 at cut A and deeper cuts (§6-7)
split/run_extraction_long.sh                      copying on LIBERO-Long (§6)
split/run_linear_record.sh, run_linear_cv.sh       linear head, 5-fold CV (§8)
split/run_linear_layers.sh, run_linear_refs.sh     depth sweep, same-scene references (§9-10)
split/run_linear_compute.sh [long24]               compute and queries needed (§11)
split/run_flops.sh                                 FLOPs per query (§11)
split/run_linear_followup.sh                       depth follow-ups and controls (§12)
split/run_long24_test30.sh, run_linear_l28_goal_object.sh   layer 24 on fresh starts, layer 28 on Goal/Object (§12)
split/run_robustness.sh                            distribution shift (§13)
split/bench_tee.py --out <json>                    what fits in a CPU TEE (§14)

# Secret beginning / middle (§17)
split/public_diff.py, run_public_swap.sh           public weights swapped in
split/run_secret_lora.sh                           public weights + LoRA
split/run_blackbox.sh                              from scratch (fit_blackbox.py)
split/run_blackbox_more.sh                         longer training, layers 1-9 (records layer 9)
split/run_secret_lora_more.sh                      LoRA on 9-16 and 1-9 (needs the layer-9 recording)
split/run_blackbox_{fill,e2e,19,seed1}.sh          end-to-end fine-tune (fit_blackbox_e2e.py), second seed
split/run_blackbox_e2e16.sh [n100]                 layers 1-16 end to end
split/run_blackbox_rel.sh                          relative loss (tried, not used)
split/diagnose_blackbox.py, attribute_blackbox.py  open-loop diagnosis
split/check_record_actions.py                      recorded actions as targets
split/run_blackbox_fresh30.sh                      end-to-end fix on fresh starts 30-39

# π0.5 (§15) and LIBERO-Plus (§16)
split/pi05_check_state.py, run_pi05_baselines.sh   checks, baselines with recording
split/ridge_pi05.py                                choosing the linear map's input
split/run_pi05_cv.sh, run_pi05_mlp_cv.sh, pi05_cv_summary.py [mlp]   5-fold linear map / MLP
split/run_liberoplus.py, liberoplus_summary.py     LIBERO-Plus, real models vs. readouts
```

Comparisons always pair a replacement with a real-model run that started at the same episode: LIBERO's scenes depend
on episode order (RESULTS.md §9).

<details><summary>Single runs by hand (from <code>openvla-oft/</code>; one RTX 4090)</summary>

Real models, 500 episodes per suite (~1–2 h):

```bash
source env.sh && vla_activate && cd openvla-oft
S=libero_spatial                       # or libero_object, libero_goal, libero_10
# OpenVLA-OFT, unsplit (stock script)
python experiments/robot/libero/run_libero_eval.py --task_suite_name $S --seed 7 \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-${S/_/-} --local_log_dir $VLA_LOGS
# OpenVLA-OFT, split at cut A: export Model 2 once, then Model 1 in-process + Model 2 over a socket
python ../split/export_model2.py --suite $S
python ../split/run_split_eval.py --suite $S
# π0.5 (starts its server in openpi's env itself)
python ../split/run_pi05_eval.py --suite $S
```

A readout, fold 0 (the 5-fold scripts above do all folds with same-scene references):

```bash
# OFT: record (interface, action) pairs with the real split, fit the linear map on starts 10-49, test on starts 0-9
python ../split/run_split_eval.py --suite $S --split_root $VLA_DATA/linear_head/record --bundle $VLA_SPLIT/$S/model2 --save_pairs $VLA_DATA/linear_head/pairs/$S
python ../split/ridge_head.py --pairs_dir $VLA_DATA/linear_head/pairs/$S --test_from 0 --test_to 10 --val_from 10 --val_to 20 \
    --out $VLA_DATA/linear_head/heads/${S}_fold0
python ../split/run_split_eval.py --suite $S --episodes_per_task 10 --bundle $VLA_DATA/linear_head/heads/${S}_fold0 --tag ridge_fold0

# π0.5: record backbone states with the baseline run, fit the linear map or the MLP, serve it
python ../split/run_pi05_eval.py --suite $S --save_actions --record_dir $VLA_DATA/pi05/record/$S
python ../split/ridge_pi05.py --record_dir $VLA_DATA/pi05/record/$S --layer layer18_in --pool 4 \
    --test_from 0 --test_to 10 --val_from 10 --val_to 20 --out $VLA_DATA/pi05/cv/heads/${S}_fold0
python ../split/mlp_pi05.py --record_dir $VLA_DATA/pi05/record/$S --cache $VLA_DATA/pi05/features/pool4/$S.pt \
    --test_from 0 --test_to 10 --val_from 10 --val_to 20 --out $VLA_DATA/pi05/mlp_cv_std/heads/${S}_fold0
python ../split/run_pi05_eval.py --suite $S --episodes_per_task 10 --linear $VLA_DATA/pi05/cv/heads/${S}_fold0 --tag linear_fold0
python ../split/run_pi05_eval.py --suite $S --episodes_per_task 10 --mlp $VLA_DATA/pi05/mlp_cv_std/heads/${S}_fold0 --tag mlp_fold0
```

LIBERO-Plus (needs the fold-0 readouts for all four suites; ~3 days on two GPUs, resumes from finished chunks):

```bash
source env.sh && vla_activate_plus
python split/run_liberoplus.py --dry_run           # lists the 215 jobs: 5 setups x 4 suites, 250-task chunks
python split/run_liberoplus.py --gpus 1            # or --gpu_units 1:2 0:1 for a second, partly free GPU
python split/liberoplus_summary.py                 # -> $VLA_DATA/libero_plus/summary.json
```

One chunk by hand (`--plus_prompt` sends the untagged LIBERO instruction, RESULTS.md §16):

```bash
cd openvla-oft && python ../split/run_pi05_eval.py --suite libero_spatial --episodes_per_task 1 --task_from 0 --task_to 250 \
    --plus_prompt --linear $VLA_DATA/pi05/cv/heads/libero_spatial_fold0 --out_root $VLA_DATA/libero_plus/evals/pi05_linear --tag pi05_linear
```
</details>
