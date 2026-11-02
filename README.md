# VLA Split

Splitting OpenVLA-OFT into two models that run in sequence: Model 1 (vision encoders + Llama 2 backbone) produces an intermediate representation, and Model 2 (the MLP action head) turns it into robot actions. Evaluated on the LIBERO simulation benchmark.

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: baseline, cut A split, extraction (copying) tests, linear-head study |

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
