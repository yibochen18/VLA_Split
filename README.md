# VLA Split

Splitting OpenVLA-OFT into two models that run in sequence: Model 1 (vision encoders + Llama 2 backbone) produces an intermediate representation, and Model 2 (the MLP action head) turns it into robot actions. Evaluated on the LIBERO simulation benchmark.

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results so far |

## Layout

```
split/                    our code (openvla-oft is used unmodified)
setup_and_smoke_test.sh   one-time environment setup + 10-episode smoke test
openvla-oft/, LIBERO/     external repos, not tracked (see below)
```

Model weights, logs, rollout videos and reference datasets live outside the repo under `/mnt/ssd/yibo/`.

## External dependencies

Clone these into the repo root at the pinned commits:

```bash
git clone https://github.com/moojink/openvla-oft.git && git -C openvla-oft checkout e4287e94541f459edc4feabc4e181f537cd569a8
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
bash setup_and_smoke_test.sh
```

`setup_and_smoke_test.sh` documents the package pins needed on top of the upstream instructions.
