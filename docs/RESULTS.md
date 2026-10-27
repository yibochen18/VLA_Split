# Results

Running record of measured results for the OpenVLA-OFT split project.

**Status (2026-10-27):** Step 1 (setup and baseline) is complete: all four suites reproduce the paper. Step 2 (split) has not started.

---

## Setup

| Item | Value |
|---|---|
| Hardware | 1× NVIDIA RTX 4090 (24 GB), shared server; runs on GPU 1 |
| Model | OpenVLA-OFT, one released checkpoint per suite (`moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}`) |
| Eval config | Script defaults: 2 images (third-person + wrist), proprio, L1-regression head, center crop, 8-step chunks executed open-loop |
| Software | Python 3.10, PyTorch 2.2.0 (cu121), transformers 4.40.1 (OFT fork), robosuite 1.4.1, MuJoCo 3.3.0 |
| Differences from paper | RTX 4090 vs. A100; MuJoCo 3.3.0; one seed (7) vs. 3 seeds averaged |

---

## 1. Baseline success rate (unsplit model)

500 episodes per suite (10 tasks × 50), seed 7. Standard error at ~97% is about ±0.8 pp.

| Suite | Ours | Paper | Difference | Failures |
|---|---|---|---|---|
| Spatial | **98.6%** (493/500) | 97.6% | +1.0 | 7 |
| Object | **98.2%** (491/500) | 98.4% | −0.2 | 9 |
| Goal | **97.6%** (488/500) | 97.9% | −0.3 | 12 |
| Long (LIBERO-10) | **93.8%** (469/500) | 94.5% | −0.7 | 31 |
| **Average** | **97.05%** | **97.1%** | −0.05 | 59 |

**Conclusion:** all four suites are within about 1 pp of the paper, i.e. within noise, and the averages match (97.05% vs. 97.1%). The released checkpoints reproduce on this server and software stack. **Step 1 checkpoint passed.**

### Tasks below 100% (out of 50 episodes each)

**Spatial:** 7 of 10 tasks at 100%

| Task | Success |
|---|---|
| pick up the black bowl on the ramekin and place it on the plate | 92% |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 96% |
| pick up the black bowl on the stove and place it on the plate | 98% |

**Object:** 4 of 10 tasks at 100%

| Task | Success |
|---|---|
| pick up the bbq sauce and place it in the basket | 94% |
| pick up the salad dressing and place it in the basket | 96% |
| cream cheese / tomato sauce / milk / chocolate pudding → basket | 98% each |

**Goal:** 4 of 10 tasks at 100%

| Task | Success |
|---|---|
| open the top drawer and put the bowl inside | 92% |
| put the wine bottle on top of the cabinet | 94% |
| put the bowl on the stove | 96% |
| open the middle drawer / push the plate to the front of the stove / put the wine bottle on the rack | 98% each |

**Long:** 3 of 10 tasks at 100%

| Task | Success |
|---|---|
| put both moka pots on the stove | **72%** |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 88% |
| put both the alphabet soup and the tomato sauce in the basket | 92% |
| black bowl in bottom drawer + close it / two mugs on two plates / mug in microwave + close it | 96% each |
| put both the alphabet soup and the cream cheese box in the basket | 98% |

In Spatial, Object and Goal, failures are spread thinly. Long has one clear weak spot: "put both moka pots on the stove" accounts for 14 of the 31 Long failures. It is the one task worth watching when comparing split vs. unsplit.

---

## 2. Latency per model query (unsplit)

LIBERO-Spatial, 131 queries (first excluded as warmup), RTX 4090, bf16. Timed with CUDA syncs around each module.

| Component | Mean (ms) | Share | Becomes |
|---|---|---|---|
| Vision backbone (DINOv2 + SigLIP, 2 images) | 26.6 | 19% | Model 1 |
| Projector + proprio projector | 2.1 | 2% | Model 1 |
| Llama 2 backbone (32 layers) | 93.7 | 68% | Model 1 |
| **Action head** | **1.2** | **1%** | **Model 2** |
| Preprocessing (image prep, tokenization) + un-normalization | 14.0 | 10% | Model 1 / Model 2 |
| **Total per query** | **137.6** | 100% | |

One query produces 8 actions, so the model cost per control step is about 17 ms. A full LIBERO episode takes about 6–7 s of wall-clock time, mostly CPU simulation.

---

## 3. Sizes

| Item | Value |
|---|---|
| Model 1 (VLA: vision + projector + Llama 2) | 7.54B params, 15.1 GB (bf16) |
| Proprio projector (stays in Model 1) | 16.8M params |
| **Model 2 (action head)** | **151.1M params, 302 MB (bf16)**, which is 2% of Model 1 |
| — of which `fc1` (28,672 → 4096) | ~117M params |
| — two residual blocks (4096 → 4096) + `fc2` (4096 → 7) | ~34M params |
| **Interface per query** | **1 × 56 × 4096 bf16 = 458,752 bytes (0.46 MB)** |

The cut point is in `_regression_or_discrete_prediction` in `openvla-oft/prismatic/extern/hf/modeling_prismatic.py`. `predict_action` already returns the interface tensor as its second output.

---

## 4. Determinism and head-only replay

LIBERO-Spatial, 1 episode per task (10 episodes, 132 queries), run twice with seed 7.

| Check | Result |
|---|---|
| Same number of queries in both runs | ✅ 132 = 132 |
| Same success per episode | ✅ 10/10 both runs |
| Hidden states (interface) bitwise identical | ✅ |
| Final actions bitwise identical | ✅ (max diff 0.0) |
| Head alone, re-run on saved hidden states, reproduces actions | ✅ bitwise (max diff 0.0) |

**Implications:**

- Evaluation is fully reproducible, so one run per suite is enough.
- The Step 2 split can be required to match the original **exactly**.
- The last check is an early, offline version of the split: feeding saved interface tensors to the head alone gives the original actions.

---

## Artifacts

| What | Where |
|---|---|
| Eval logs (per episode) | `/mnt/ssd/yibo/logs/EVAL-libero_*-openvla-*.txt` |
| Console logs | `/mnt/ssd/yibo/logs/baseline_*_seed7.log` |
| Reference dataset (132 queries: images, state, task, interface tensor, actions) | `/mnt/ssd/yibo/reference/libero_spatial_seed7_1ep.pt` |
| Probe summary (latency, sizes, determinism) | `/mnt/ssd/yibo/reference/libero_spatial_seed7_1ep_summary.json` |
| Rollout videos (one MP4 per episode) | `/mnt/ssd/yibo/rollouts/` |

## Reproducing

```bash
bash setup_and_smoke_test.sh                     # environment (one-time)
bash split/run_baselines.sh                      # full baselines (edit suite list inside)
cd openvla-oft && CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
    python ../split/probe_baseline.py            # reference dataset, latency, determinism
```
