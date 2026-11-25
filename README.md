# VLA Split

Splitting OpenVLA-OFT into two models that run in sequence: Model 1 (vision encoders + Llama 2 backbone) produces an intermediate representation, and Model 2 (the MLP action head) turns it into robot actions. Evaluated on the LIBERO simulation benchmark. The linear-replacement study is repeated on π0.5, whose action module is a flow-matching transformer rather than an MLP head.

Main findings: the split is exact; Model 2 can be copied from ~100 observed queries; one linear map fitted from observed traffic replaces Model 2 with no loss in success or robustness, and from layer 24 also replaces the top 8 Llama layers (~15% faster per query). On π0.5, one linear map replaces the ≈430M-parameter action expert with no significant loss overall on LIBERO (96.0% vs. 96.7%), with queries 73% faster in uncompiled PyTorch; may be smaller in a compiled deployment; a small MLP does about as well. Off-distribution (LIBERO-Plus, 10,030 perturbed tasks, §16), readouts fitted on clean LIBERO behave differently for the two models: OFT's linear map stays within ~1 pp of the real head (−1.1 pp), while π0.5's linear map and MLP lose 14.8 and 9.7 pp to the real expert (up to −42 pp under camera shifts). Keeping the beginning or middle of OFT secret instead (§17) fails too: OFT is the public `openvla-7b` plus a rank-32 LoRA, so public weights plus a LoRA fitted on ~100–1,000 observed queries of the secret part's input and output restore the robot to within ~3 points of the real model; without public weights, a small transformer trained from scratch on ~1,000–5,000 observed queries still replaces a secret middle block (layers 9–16 at 5,000 queries: 97–99 on Spatial, 91–93 on Long over two training seeds; real 95 / 94). Blocks starting at layer 1 are unreliable when trained only to reproduce the block's output (layers 1–9: 11–92; layers 1–16 at 5,000: Spatial 84–91, Long 63–80); fine-tuned end to end through the attacker's own later layers to match the executed actions, layers 1–9 score 97–99 on Spatial with ≥ 1,000 queries, and at 5,000 queries layers 1–9 and 1–16 reach 97–100 on Spatial and 79–95 on Long (two seeds).

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: baseline, cut A split, extraction tests, linear-head replacement, depth sweep, compute and observation cost, robustness, CPU TEE budget, π0.5 (linear map and MLP), LIBERO-Plus robustness, secret beginning or middle layers |

## Layout

```
env.sh                    machine-specific settings: data location, conda env, GPU (single source of truth)
split/                    our code (openvla-oft and openpi are used unmodified)
setup_and_smoke_test.sh   one-time environment setup + 10-episode smoke test
openvla-oft/, LIBERO/     external repos, not tracked (see below)
openpi/                   π0.5 (Physical Intelligence), not tracked; runs in its own environment
LIBERO-plus/              LIBERO-Plus benchmark, not tracked; its own conda env (see below)
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

For LIBERO-Plus (§16), a clone of the OFT conda env with LIBERO-Plus installed in place of LIBERO; `vla_activate_plus` (from `env.sh`) activates it and points LIBERO's config at `$VLA_DATA/libero_plus/config`, so `~/.libero` and the original env are untouched. Its `requirements.txt` is not installed (it would downgrade numpy, transformers and robosuite):

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

## Experiments

Each script writes only under `$VLA_DATA`; run them in this order (RESULTS.md section in brackets):

```
split/run_baselines.sh, split/run_v3_all.sh      baselines and the bitwise split check (§1-5)
split/run_extraction.sh, split/run_layers.sh     copying Model 2 at cut A and cuts B-k (§6-7)
split/run_extraction_long.sh                    the cut-A copying test on LIBERO-Long (§6)
split/run_linear_record.sh, run_linear_cv.sh     linear head, 5-fold cross-validation (§8)
split/run_linear_layers.sh, run_linear_refs.sh   depth sweep and same-scene references (§9-10)
split/run_linear_compute.sh                      compute saved and observation needed (§11)
split/run_flops.sh                               FLOPs per query, OFT and π0.5 (§11)
split/run_linear_followup.sh                     limits of depth and controls (§12)
split/run_long24_test30.sh                       Long, layer 24, on starts 30-39 (§12)
split/run_robustness.sh                          robustness under distribution shift (§13)
split/bench_tee.py                               what fits in a CPU TEE (§14)
split/pi05_check_state.py, run_pi05_baselines.sh π0.5: checks and baselines with recording (§15)
split/ridge_pi05.py                              π0.5: choosing the linear map's input (§15)
split/run_pi05_cv.sh, pi05_cv_summary.py         π0.5: 5-fold closed loop vs. same-scene π0.5 (§15)
split/run_pi05_mlp_cv.sh, pi05_cv_summary.py mlp π0.5: small MLP in place of the expert, 5-fold (§15)
split/run_liberoplus.py, liberoplus_summary.py   LIBERO-Plus, real models vs. clean-LIBERO readouts (§16)
split/public_diff.py, run_public_swap.sh         secret beginning/middle: weight comparison, public weights swapped in (§17)
split/run_secret_lora.sh, run_secret_lora_more.sh  secret beginning/middle: LoRA fitted on recorded traffic (§17)
split/run_blackbox.sh, fit_blackbox.py           secret block, no public weights: replacements trained from scratch (§17)
split/fit_blackbox_e2e.py                        black box: fine-tune a replacement end to end through the later layers (§17)
split/diagnose_blackbox.py, attribute_blackbox.py  black box: open-loop action error, error by token group (§17)
split/check_record_actions.py                    recorded executed actions vs. the real model replaying them (§17)
split/run_blackbox_{more,fill,e2e,19,e2e16,seed1}.sh  black-box follow-up: layers 1–9, end to end, second seed (§17)
split/run_blackbox_fresh30.sh                    end-to-end fix on fresh test starts 30-39 (§17)
```

## Running on LIBERO and LIBERO-Plus

Every evaluation runs from `openvla-oft/` (its LIBERO harness drives all episodes) and writes a result JSON plus a log
under `$VLA_DATA`. Times are for one RTX 4090.

**1. Real models on LIBERO** (500 episodes per suite: 10 tasks × 50 initial states, seed 7; ~1–2 h per suite)

```bash
source env.sh && vla_activate && cd openvla-oft
S=libero_spatial                       # or libero_object, libero_goal, libero_10
# OpenVLA-OFT, unsplit (stock script)
python experiments/robot/libero/run_libero_eval.py --task_suite_name $S --seed 7 \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-${S/_/-}
# OpenVLA-OFT, split at cut A: export Model 2 once, then Model 1 in-process + Model 2 over a socket
python ../split/export_model2.py --suite $S
python ../split/run_split_eval.py --suite $S
# π0.5 (starts its server in openpi's env itself)
python ../split/run_pi05_eval.py --suite $S
```

**2. Readouts on LIBERO** (record traffic, fit, then serve the readout in place of the action module)

```bash
# OFT: record (interface, action) pairs with the real split, fit the linear map on starts 10-49, test on starts 0-9
python ../split/run_split_eval.py --suite $S --split_root $VLA_DATA/linear_head/record --save_pairs $VLA_DATA/linear_head/pairs/$S
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

This is fold 0. The full 5-fold runs with their paired same-scene references are `run_linear_record.sh` +
`run_linear_cv.sh` (OFT) and `run_pi05_baselines.sh` + `run_pi05_cv.sh` + `run_pi05_mlp_cv.sh` (π0.5); summaries
with `pi05_cv_summary.py [mlp]`. A readout's real-model reference must be run the same way (same first start), since
LIBERO's scenes depend on episode order (RESULTS.md §9).

**3. LIBERO-Plus** (needs step 2's fold-0 readouts for all four suites; install as above)

```bash
source env.sh && vla_activate_plus
python split/run_liberoplus.py --dry_run           # lists the 215 jobs: 5 setups x 4 suites, 250-task chunks
python split/run_liberoplus.py --gpus 1            # or --gpu_units 1:2 0:1 for a second, partly free GPU
python split/liberoplus_summary.py                 # -> $VLA_DATA/libero_plus/summary.json
```

All 50,150 episodes took ~3 days on two shared RTX 4090s; rerunning resumes from the finished chunks. One chunk by
hand, e.g. π0.5 with its linear map on Spatial tasks 0-249:

```bash
cd openvla-oft && python ../split/run_pi05_eval.py --suite libero_spatial --episodes_per_task 1 --task_from 0 --task_to 250 \
    --plus_prompt --linear $VLA_DATA/pi05/cv/heads/libero_spatial_fold0 --out_root $VLA_DATA/libero_plus/evals/pi05_linear --tag pi05_linear
```

`--plus_prompt` sends each task's untagged LIBERO instruction (`split/liberoplus_prompts.py`, RESULTS.md §16); leave
it out to use LIBERO-Plus's own (tagged) `task.language`.
