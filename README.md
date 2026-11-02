# VLA Split

Splitting OpenVLA-OFT into two models that run in sequence: Model 1 (vision encoders + Llama 2 backbone) produces an intermediate representation, and Model 2 (the MLP action head) turns it into robot actions. Evaluated on the LIBERO simulation benchmark.

Main findings: the split is exact; Model 2 can be copied from ~100 observed queries; one linear map fitted from observed traffic replaces Model 2 with no loss in success or robustness, and from layer 24 also replaces the top 8 Llama layers (~15% faster per query).

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: baseline, cut A split, extraction tests, linear-head replacement, depth sweep, compute and observation cost, robustness |

## Layout

```
env.sh                    machine-specific settings: data location, conda env, GPU (single source of truth)
split/                    our code (openvla-oft is used unmodified)
setup_and_smoke_test.sh   one-time environment setup + 10-episode smoke test
openvla-oft/, LIBERO/     external repos, not tracked (see below)
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
```
