# Results

Running record of measured results for the OpenVLA-OFT split project.

**Status (2026-11-02):** Baseline reproduced on all four suites (97.05% vs. 97.1%). Cut A split verified bitwise on all four suites, and full 500-episode split runs match the unsplit model exactly on every suite. Extraction tests (§6, §7): Model 2 can be copied from ~100 observed queries, even with up to 4 Llama layers moved into it. Linear-head study (§8): a single linear map distilled from observed traffic replaces Model 2 with no measurable loss in task success (97.15% vs. 96.85% for the MLP on identical scenes, 2,000 episodes), and is as robust as the real model under camera, lighting, noise and object-position shifts (§13). LIBERO's scenes depend on episode order (§9), so every comparison uses a real-model run done the same way. Depth (§10–§12): a linear map reading Llama layer 24 replaces the top 8 layers plus the head with small loss on all four suites (~15% faster); layer 20 (23% faster, 32% less memory) holds on Spatial but loses ~2–10 points on the other suites and is clearly less robust to camera and object shifts (§13). A small MLP readout recovers layers 12–16, so the fall-off there is the linear map's limit; at layers 4–8 the information is not yet there. Observation needed: ~100 queries for the head alone, ~2,500 for layer 20 on Spatial (§11).

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

Paths are relative to `$VLA_DATA` (set in `env.sh`; `/mnt/ssd/yibo` on our server).

| What | Where |
|---|---|
| Eval logs (per episode) | `logs/EVAL-libero_*-openvla-*.txt` |
| Console logs | `logs/baseline_*_seed7.log` |
| Reference dataset (132 queries: images, state, task, interface tensor, actions) | `reference/libero_spatial_seed7_1ep.pt` |
| Probe summary (latency, sizes, determinism) | `reference/libero_spatial_seed7_1ep_summary.json` |
| Rollout videos (one MP4 per episode) | `rollouts/` |

## Reproducing

```bash
# edit env.sh first if not on our server
bash setup_and_smoke_test.sh                     # environment (one-time)
bash split/run_baselines.sh                      # full baselines (edit suite list inside)
source env.sh && vla_activate && cd openvla-oft
python ../split/probe_baseline.py            # reference dataset, latency, determinism
```

---

## 5. Cut A split (Model 1 = backbone, Model 2 = action head)

Model 2 contains the action head, action normalization stats and gripper post-processing, and runs as a separate server process on the same GPU, connected by a Unix socket. Model 1 never loads the head and has no action stats.

### Equivalence (bitwise, against the unsplit model)

| Check | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| V0: Model 2 loads without any openvla-oft / transformers code | ✅ | ✅ | ✅ | ✅ |
| V1a/V1b/V2: each half, then two processes, on saved queries | ✅ 132/132 | — | — | — |
| V3: closed-loop episodes (1 per task), interface tensors + actions bitwise equal | ✅ 132/132 | ✅ 187/187 | ✅ 133/133 | ✅ 332/332 |
| V3: same successes as unsplit | ✅ 10/10 | ✅ 10/10 | ✅ 10/10 | ✅ 10/10 |

Max absolute difference in every comparison: **0.0**.

### Latency per query (ms, mean, first query excluded)

Both measured end to end with no per-module timers, Spatial, the same 10 episodes:

| | Unsplit | Split |
|---|---|---|
| Model 1 (preprocessing + vision + Llama) | — | 101.0 |
| Serialize (GPU→CPU + bytes) | — | 0.17 |
| Socket transfer | — | 0.29 |
| Model 2 server (decode + head + un-norm + gripper + encode) | — | 0.84 |
| **Total** | **101.9** | **102.3** |

**Split overhead: +0.35 ms per query (+0.3%).** Other suites: split total 102.2–102.3 ms.

### Interface and Model 2 size

| Item | Value |
|---|---|
| Request per query | 458,784 bytes (32-byte header + 1×56×4096 bf16) |
| Response per query | 472 bytes (24-byte header + 8×7 float64 actions) |
| Model 2 params | 151.1M (302 MB bf16; `head.pt` on disk 302 MB) |
| Model 2 bundles | `$VLA_DATA/split/<suite>/model2/` |

### Full 500-episode split evaluation (Spatial)

**493/500 (98.6%)**, identical to the unsplit baseline, with the same per-task results. Mean total latency 101.8 ms per query.

---

## 6. Extraction (copying) test, cut A, Spatial

Can someone who sees the cut-A traffic (interface tensor in, executed actions out) build their own Model 2? We recorded the real split's traffic over 500 episodes (~6,600 queries), trained replacement heads on N recorded queries from episodes 0–39, and ran each replacement **in place of Model 2** on 100 held-out starts.

### Closed-loop success (the robot actually doing the task)

| Queries recorded | ≈ robot time | `oft` (151M) | `mlp` (14.7M) | `linear` (0.2M) |
|---|---|---|---|---|
| 100 | 8 episodes, ~40 s | 97 | 96 | 81 |
| 500 | 39 episodes, ~3 min | 97 | 98 | 92 |
| 1,000 | 76 episodes, ~6 min | 96 | 98 | 98 |
| 2,500 | 190 episodes, ~15 min | 97 | 99 | 97 |
| 5,000 | 378 episodes, ~30 min | 98 | 98 | 96 |

Closed-loop successes out of the 100 held-out episodes (initial states 40–49 of every task, run alone starting at 40). **The real Model 2 scores 95/100 when run the same way** (98/100 on the same starts inside its full 500-episode run, where the scenes differ slightly, §9). With 100 episodes, differences of 1–3 are within noise (±~3 at these rates).

### Offline action error on held-out queries

| Queries | `oft` L1 | `mlp` L1 | `linear` L1 |
|---|---|---|---|
| 100 | 0.0107 | 0.0202 | 0.0535 |
| 500 | 0.0042 | 0.0087 | 0.0351 |
| 1,000 | 0.0035 | 0.0067 | 0.0288 |
| 2,500 | 0.0028 | 0.0054 | 0.0215 |
| 5,000 | 0.0026 | 0.0047 | 0.0185 |

Held-out mean L1 error on the 6 continuous action dimensions (always predicting the mean: 0.195). Gripper accuracy ≥ 99.96% for every replacement. Training time per replacement: 2–3 s (`linear`), 8 s (`mlp`), 68 s (`oft`) on one GPU.

### Takeaway

**Cut A gives essentially no protection against copying.** A replacement trained for ~10 seconds on 100 recorded queries (under a minute of robot operation) matches the real Model 2 (97 vs 95 out of 100 on identical scenes). Even a single linear layer reaches the real model's level from 1,000 queries. Hiding the action head behind the interface is therefore not enough to keep Model 2's behavior secret; the next step is a deeper cut (B-k) and the same measurement.

Data: `$VLA_DATA/extraction/libero_spatial/` (pairs, replacements), closed-loop results `$VLA_DATA/split/libero_spatial/split_eval_seed7_10ep_from40_<arch>_n<N>.json`.

---

## 7. Layer-depth extraction test (would a deeper cut help?), Spatial

Same test as §6, but the copier sees the 56 action positions after Llama layer 28, 30 or 31 instead of the cut A tensor. That is what a cut B-k (k = 4, 2, 1 layers moved into Model 2) would expose at those positions; a real B-k interface would also expose every other token, so this is the attacker's *minimum* information. Recorded during a 500-episode run of the real split (493/500 again, identical per episode); Model 1 reads the layers from hidden states it already computes, and cut A's bitwise check (V3) still passes with recording on.

Closed-loop successes out of the 100 held-out episodes; **the real Model 2 scores 95/100 on the same scenes** (98 inside its full run, §9). "Layer 32" is cut A (from §6).

| Copy type | Queries | Layer 28 (B-4) | Layer 30 (B-2) | Layer 31 (B-1) | Layer 32 (cut A) |
|---|---|---|---|---|---|
| `oft` (151M) | 100 | 96 | 98 | 96 | 97 |
| | 1,000 | 100 | 99 | 98 | 96 |
| | 5,000 | 98 | 97 | 98 | 98 |
| `mlp` (14.7M) | 100 | 94 | 90 | 91 | 96 |
| | 1,000 | 98 | 99 | 98 | 98 |
| | 5,000 | 97 | 98 | 96 | 98 |
| `linear` (0.2M) | 100 | 4 | 8 | 11 | 81 |
| | 1,000 | 81 | 76 | 66 | 98 |
| | 5,000 | 97 | 96 | 95 | 96 |

Model 2 size if built at that cut (bf16): B-4 ~1.9 GB, B-2 ~1.1 GB, B-1 ~0.7 GB, cut A 0.3 GB.

**Moving up to 4 layers into Model 2 does not make it meaningfully harder to copy.** At every cut, a copy with OFT's head architecture trained on 100 recorded queries (under a minute of robot time, ~1 minute of training) scores 96–98, at or above the real Model 2 (95). The only copy that gets worse at deeper cuts is the bare linear layer, and it still reaches 95–97 at 5,000 queries. Its early weakness is most likely because intermediate Llama states are unnormalized, with values up to ~200, and the linear copy has no LayerNorm; the `mlp` and `oft` copies do, and are barely affected. A real attacker would normalize, so this is not protection.

Conclusion: within any reasonable Model 2 size budget, choosing the cut layer does not keep Model 2's behavior secret. The interface itself has to change (for example, revealing less than the full hidden state), or Model 2 must be protected by something other than the cut location.

Data: `$VLA_DATA/extraction/libero_spatial_layers/` (pairs per layer, replacements), closed-loop results `$VLA_DATA/split_layers/libero_spatial/`.

---

## 8. Replacing Model 2 with a single linear map fitted from observed traffic

Question: given a working model whose Model 2 is OFT's MLP head, can we observe its inputs (the cut A interface tensor) and outputs (the executed actions) and replace **all of Model 2** (MLP head, action un-normalization, gripper rule) with one linear map?

**Replacement:** per chunk step, `y = W·x + b` with `x` the 7 action-slot hidden states (7 × 4096 = 28,672 values) and `W` 7 × 28,672, shared across the 8 steps; gripper = sign of output 7. 200,711 parameters (0.8 MB fp32) vs 151.1M (302 MB bf16) for the MLP head. Fitted in closed form by ridge regression on standardized features (standardization folded into `W`), λ chosen on a validation fold (fit on 30 starts, refit on 40); ~70–110 s per fit on one GPU (`split/ridge_head.py`).

**Protocol:** the real split (MLP head) ran all 500 episodes of every suite while every (interface, actions) pair was recorded; these runs reproduce the unsplit baselines exactly (493 / 491 / 488 / 469). 5-fold cross-validation over initial states: for fold f, starts 10f–10f+9 of every task are held out, the linear map is fitted on the other 40, and it is served **in place of Model 2** on the 100 held-out episodes. Every episode is tested exactly once (`split/run_linear_record.sh`, `split/run_linear_cv.sh`).

| Suite | Linear map | MLP head (same scenes) | Episodes only linear fails / only MLP fails | Observed queries per fit | Held-out action error (L1) |
|---|---|---|---|---|---|
| Spatial | **493/500 (98.6%)** | 490/500 (98.0%) | 3 / 6 | ~5,300 | 0.0106 |
| Object | **489/500 (97.8%)** | 493/500 (98.6%) | 7 / 3 | ~7,100 | 0.0094 |
| Goal | **491/500 (98.2%)** | 490/500 (98.0%) | 7 / 8 | ~5,800 | 0.0080 |
| Long (LIBERO-10) | **470/500 (94.0%)** | 464/500 (92.8%) | 18 / 24 | ~13,600 | 0.0076 |
| **Average** | **97.15%** | 96.85% | 35 / 41 | | |

Gripper agreement with the MLP ≥ 99.9% on every fold. Always predicting the mean action gives L1 0.13–0.20.

The MLP side is the real MLP run exactly like each linear fold (alone, starting at start 10f), so both see identical scenes (§9); fold 0 comes from the full run, which also starts there (`split/run_linear_refs.sh`). Paired comparison: exact McNemar test on the episodes where exactly one fails; 95% CI of the difference, normal approximation on the paired differences.

| Suite | Linear − MLP | 95% CI | McNemar p |
|---|---|---|---|
| Spatial | +0.6 pp | [−0.6, +1.8] | 0.51 |
| Object | −0.8 pp | [−2.0, +0.4] | 0.34 |
| Goal | +0.2 pp | [−1.3, +1.7] | 1.00 |
| Long | +1.2 pp | [−1.3, +3.7] | 0.44 |
| **All 2,000** | **+0.3 pp** | **[−0.6, +1.2]** | **0.57** |

History: the first version of this table paired the linear folds with the MLP's full 500-episode run (+0.1 pp, [−0.7, +0.9]). That pairing assumed identical scenes, which holds only for fold 0 (§9), so it was replaced by these same-scene runs. The MLP's own scores moved by up to 5 per 100 between the two kinds of run.

**Sanity checks:** the server reported 200,711 parameters in every closed-loop run (the linear map, not the MLP); each fitted head's training data excludes its test starts (train + test = all recorded queries); the linear map's actions equal the MLP's exactly in 0 of 1,330 held-out Spatial queries (it is a different policy, ~5% relative error on movement), and linear and MLP succeed on different episodes. Weak heads fail in the same harness (a linear map fitted on only 100 queries: 81/100; at deeper layers: 4–11/100).

**Takeaway:** after observing ~5,000–14,000 queries, a single linear map replaces Model 2 with no measurable loss in task success on any of the four suites (overall +0.3 pp, 95% CI [−0.6, +1.2], on identical scenes). The MLP head is effectively redundant given OFT's fine-tuned features. Caveats: the linear map is fitted to an existing model with an MLP head (Model 1's features were trained jointly with it); its actions are less precise (error 0.008–0.011 vs 0.0026 for a re-trained copy with OFT's head architecture on Spatial), which the tasks tolerate; one seed; the head is ~1% of query latency, so the gain is size/memory, not speed.

---

## 9. Scene variation in LIBERO: an episode's scene depends on which episodes ran before it

LIBERO's fixed initial states set the robot and movable objects only. On every `env.reset()`, fixtures (cabinet, stove, ...) are re-placed by a random sampler that draws from numpy's global RNG (`LIBERO/libero/libero/envs/bddl_base_domain.py`, fixture branch of `_reset_internal`), and that RNG is reseeded once per task when the env is created (`env.seed(0)` in OFT's `get_libero_env`). So the k-th reset in a task always gives the same fixture placement, and an episode's scene depends on how many episodes ran before it in that task. Robot motion in between does not matter (checked: same reset count after 0 or 150 random steps, identical images). Runs are otherwise exactly reproducible: re-recording the full Spatial run from start 0 reproduced the original's tensors and actions bit for bit (69 of 69 episodes compared) and the same per-episode outcomes (493/500); the Long re-recording also matched its original per episode (469/500).

**Determinism check** (`split/run_linear_layers.sh`, first step): the real MLP on Spatial starts 40–49, run alone from start 40, scores **95/100**, against 98/100 for the same starts inside the full 500-episode run. All 100 episodes differ from the first query, because the cabinet and stove sit a few pixels away (physics state identical, ~8,000 of 65,536 image pixels differ). 5 outcomes flip (task 5: 4, task 7: 1). This is the benchmark's own noise level for one model on fixed starts.

**Consequences:**
- **Like-for-like comparisons need the same starting episode.** Every held-out test (§6, §7, depth sweep, linear fold 4) ran alone from start 40, so its reference is the real model run the same way: 95/100, not 98. Runs that start at the same episode see identical scenes whatever head they use.
- **§8's MLP side** was first the full run (identical scenes on fold 0 only). `split/run_linear_refs.sh` reran the real MLP on folds 1–4 of every suite the way the linear heads were run; §8 now pairs on identical scenes.
- Single 100-episode differences of ~3 are within this noise; conclusions that rest on them (none so far) would need repeats.

Data: `$VLA_DATA/linear_layers/checks/` (the check, with its recorded pairs), `$VLA_DATA/linear_head/mlp_refs/` (like-for-like references).

---

## 10. Depth sweep: how much of the network can one linear map replace?

At layer L, Model 1 keeps Llama layers 1–L, and one ridge linear map fitted on the 56 action positions after layer L (no final norm) replaces everything after it: layers L+1–32, the final norm, the MLP head and post-processing. Fitted on starts 0–39 (λ on 0–9), scored in closed loop on held-out starts 40–49, run alone from start 40, so the real model's same-scene reference is 95 (Spatial) and 94 (Long). Spatial every 2 layers down to 2; Long at 28/24/20/16/12/8 (only where Spatial scored ≥ 50). `split/run_linear_layers.sh`, ~5,300 (Spatial) / ~13,600 (Long) observed queries per fit.

| Layer read | Llama layers replaced | Params replaced | Spatial | Long | Spatial action L1 | Long action L1 |
|---|---|---|---|---|---|---|
| 32 (head only, §8 fold 4) | 0 | 0.15B | 97 | 93 | 0.0105 | 0.0076 |
| 31 | 1 | 0.35B | 96 | | 0.0118 | |
| 30 | 2 | 0.56B | 96 | | 0.0127 | |
| 28 | 4 | 0.96B | 97 | 93 | 0.0148 | 0.0124 |
| 26 | 6 | 1.37B | 97 | | 0.0165 | |
| 24 | 8 | 1.77B | 92 | 91 | 0.0197 | 0.0168 |
| 22 | 10 | 2.17B | 98 | | 0.0233 | |
| 20 | 12 | 2.58B | 96 | 92 | 0.0284 | 0.0270 |
| 18 | 14 | 2.98B | 90 | | 0.0299 | |
| 16 | 16 | 3.39B | 95 | 82 | 0.0319 | 0.0327 |
| 14 | 18 | 3.79B | 91 | | 0.0338 | |
| 12 | 20 | 4.20B | 87 | 79 | 0.0366 | 0.0372 |
| 10 | 22 | 4.60B | 81 | | 0.0385 | |
| 8 | 24 | 5.01B | 64 | 56 | 0.0416 | 0.0412 |
| 6 | 26 | 5.41B | 66 | | 0.0445 | |
| 4 | 28 | 5.82B | 59 | | 0.0469 | |
| 2 | 30 | 6.22B | 82 | | 0.0485 | |

Successes out of 100. Always predicting the mean action gives L1 0.195 (Spatial).

**Takeaways:**
- **No cliff.** Down to layer 20 (12 layers + head, ~2.6B parameters replaced), one linear map stays within noise of the real model on these starts for both suites. Spatial holds to about layer 16; Long, the harder suite, drops at 16 (82). (§12 qualifies this: on other starts Long at layer 20 scores 84 vs 94, and Goal/Object lose ~10 points at layer 20; layer 24 is the depth that holds on all four suites.)
- **Gradual fade below:** ~80–87 at layers 10–12, ~55–65 at layer 8 and below.
- **Action error rises steadily with depth** on both suites, so the success curve tracks a real loss of precision.
- **Unexplained point:** Spatial layer 2 scores 82, above layers 4–8 (59–66), although its action error is the worst. Per task, tasks 1 and 7 fail at layers 4–8 (1–4/10) but largely succeed at layer 2 (10/10 and 5/10). Repeated on other starts (§12: 79 at layer 2 vs 55 at layer 4), so it is a real effect; still not understood.
- **Caveat:** Model 1 still ran all 32 layers in these tests (the map reads intermediate hidden states), so this measures what each layer holds, not wall-clock savings. One seed, 100 episodes per point (±~3–5 noise).

Data: `$VLA_DATA/linear_layers/` (pairs per layer, heads, evals, records, checks).

---

## 11. Linear replacement: compute saved and observation needed

`split/run_linear_compute.sh`. Two questions about the §10 replacement (Model 1 runs Llama layers 1–L, one linear map does the rest): what does it actually save, and how many observed queries does the fit need?

### Compute

Model 1 can now stop after layer L (`model1.py` `depth`, `run_split_eval.py --truncate`): later layers and the final norm are removed and their weights freed. At every depth tested (32 to 8), the truncated Model 1 produces **bitwise the same tensor** the depth sweep read from a full forward pass (10 real observations per suite, Spatial and Long), so success rates carry over unchanged. The closed-loop runs confirm it: Spatial at layers 28/24/20/16 scores 97/92/96/95, exactly the sweep's numbers.

Model 1 per query (`split/bench_depth.py`: median of 200 calls on real observations, depths interleaved; weights counted from the loaded parameters):

| Llama layers kept | Spatial (ms) | Long (ms) | Model 1 weights (GB, bf16) |
|---|---|---|---|
| 32 (full) | 99.1 | 98.0 | 14.1 |
| 28 | 90.9 | 89.6 | 12.6 |
| 24 | 82.5 | 81.4 | 11.1 |
| 20 | 74.4 | 73.2 | 9.6 |
| 16 | 66.1 | 64.9 | 8.0 |
| 12 | 57.8 | 56.7 | 6.5 |
| 8 | 49.5 | 48.5 | 5.0 |
| 0 (no Llama layers) | 33.0 | 32.0 | 2.0 |

Each Llama layer costs ~2.06 ms and 0.39 GB. About 32–33 ms is fixed cost that no Llama layer removal touches (vision encoders, projector, preprocessing, output layer), which caps the possible saving at about two-thirds.

In closed loop (Spatial, 100 episodes each; the shared GPU adds ~1 ms of noise):

| Model 1 stops at | Success | Total per query | Model 1 GPU memory, peak |
|---|---|---|---|
| 32 + linear map (cut A) | 97 | 100.6 ms | 14.7 GB |
| 28 | 97 | 94.7 ms | 13.1 GB |
| 24 | 92 | 85.7 ms | 11.6 GB |
| **20** | **96** | **77.0 ms (−23%)** | **10.0 GB (−32%)** |
| 16 | 95 | 68.9 ms (−32%) | 8.4 GB (−43%) |

Real model on these scenes: 95.

Model 2 alone, per query (median):

| | GPU | One CPU thread |
|---|---|---|
| MLP head (151M) | 0.49 ms | 537 ms |
| Linear map (0.2M) | 0.11 ms | 0.16 ms |

Replacing only the head saves almost nothing on the GPU (0.4 ms of ~100 ms); the saving comes from dropping Llama layers. On a single CPU core, a stand-in for a TEE, the MLP head is far too slow for real-time control and the linear map is ~3,400× faster.

### Observed queries needed

The linear map is refitted on only N observed queries: whole episodes from starts 0–39, taken in a fixed shuffled order, with λ chosen on ~20% of those episodes (`ridge_head.py --max_queries`). It is then scored in closed loop on starts 40–49, with Model 1 truncated at the same layer. Successes out of 100; real model 95 (Spatial) and 94 (Long) on these scenes. "All" is §10 (~5,300 Spatial / ~13,600 Long queries, λ on starts 0–9).

| Spatial | 100 | 500 | 1,000 | 2,500 | All |
|---|---|---|---|---|---|
| Layer 32 (head only) | 95 | 96 | 97 | 97 | 97 |
| Layer 24 | 78 | 93 | 96 | 94 | 92 |
| Layer 20 | 74 | 89 | 89 | 92 | 96 |
| Layer 16 | 62 | 83 | 83 | 93 | 95 |

| Long | 500 | 1,000 | 2,500 | 5,000 | All |
|---|---|---|---|---|---|
| Layer 32 (head only) | 87 | 94 | 94 | 91 | 93 |
| Layer 20 | 67 | 78 | 84 | 86 | 92 |

Robot time: 100 queries ≈ 8 Spatial episodes (~40 s), 2,500 ≈ 190 episodes (~15 min).

**Takeaways:**
- Replacing only the head is extremely cheap: 100 observed queries already match the real model on Spatial.
- Replacing more needs more observation. At 100 queries, layers 24/20/16 score 78/74/62; at 2,500 queries, all are within a few points of the real model.
- Long needs several times more data: layer 20 is still at 86 after 5,000 queries and reaches 92 only with all ~13,600.
- Overall: on Spatial, watching the model for roughly 15–30 minutes of operation is enough to replace the top 12 Llama layers plus the head with one linear map that makes each query 23% faster and Model 1 32% smaller. Across suites and under distribution shift, layer 24 (8 layers, ~15% faster) is the safer depth (§12, §13).

**Caveats:** one seed per point (±~3–5 per 100 episodes); a second subset draw (§12) moves results by up to 13 at 100 queries but ≤ 4 from 500; at small N some tasks may be missing from the observed episodes; latencies are on an RTX 4090 shared with other users' jobs (interleaving keeps the comparison fair).

**Also found:** OpenVLA-OFT's `get_vla_action` normalizes the observation's robot state in place, so calling it twice on the same observation dict changes the input. Evaluations build a fresh observation every step, so no result is affected; `bench_depth.py` copies each observation before every call.

Data: `$VLA_DATA/linear_compute/` (`bench/`, `evals/`, `heads/`).

---

## 12. Follow-ups: what limits depth, and do the depth results hold up?

`split/run_linear_followup.sh` (2026-11-01/02). Successes out of 100 held-out episodes. Every reference is the real model run the same way: same first episode, so identical scenes.

### Small MLP readout vs. linear map at deep layers

LayerNorm → 512 → 7 (14.7M parameters, `train_replacement.py --arch mlp`), trained on the same layer-L pairs (5,000 Spatial / 13,000 Long queries, starts 0–39), scored on starts 40–49 with Model 1 truncated at layer L.

| Layer | Spatial: linear / MLP | Long: linear / MLP | Spatial held-out L1: linear / MLP |
|---|---|---|---|
| 16 | 95 / **97** | 82 / **91** | 0.032 / 0.025 |
| 12 | 87 / **92** | 79 / **84** | 0.037 / 0.029 |
| 8 | 64 / 77 | 56 / 57 | 0.042 / 0.035 |
| 4 | 59 / 53 | – | 0.047 / 0.040 |

Real model: 95 (Spatial), 94 (Long).

A small nonlinear readout recovers layers 12–16 (Long 82 → 91 at layer 16), so the linear map's fall-off there is a limit of linearity, not of the information available. At layers 4–8 neither readout works: the action-relevant information is not yet in the action positions.

### Replicate on different episodes (test starts 30–39)

Refit without starts 30–39 (λ on 40–49), tested on 30–39.

| Layer | 24 | 20 | 16 | 12 | 8 | 4 | 2 |
|---|---|---|---|---|---|---|---|
| Spatial, starts 30–39 (real 100) | 99 | 98 | 99 | 85 | 67 | 55 | 79 |
| Spatial, starts 40–49 (real 95, §10) | 92 | 96 | 95 | 87 | 64 | 59 | 82 |
| Long, starts 30–39 (real 94) | | 84 | 78 | | | | |
| Long, starts 40–49 (real 94, §10) | | 92 | 82 | | | | |

- **Spatial replicates:** within noise of the real model down to layer 16, then the same fade.
- **The layer-2 anomaly repeats** (79 vs 55 at layer 4).
- **Long at layer 20 does not hold on these starts** (84 vs 94), so the earlier "Long holds to layer 20" was partly luck of the test set.

### Goal and Object

Recorded with `--record_layers 24 20 16`; the recording runs reproduced the baselines (488/500, 491/500). Fitted as in §10.

| Layer | Goal (real 100) | Object (real 99) | Goal / Object held-out L1 |
|---|---|---|---|
| 24 | 95 | 97 | 0.018 / 0.017 |
| 20 | 89 | 90 | 0.026 / 0.025 |
| 16 | 86 | 93 | 0.032 / 0.031 |

Layer 24 is within a few points of the real model on both; layer 20 loses ~10.

### Controls

- **Constant action** (W = 0, b = mean observed action; what any linear map reading the layer-0 action positions, which are all zeros, reduces to): **0/100** on Spatial. Success requires information from the model.
- **Second data subset** (`--max_queries` with seed 1, Spatial). Seed 0 → seed 1:

| Queries | Layer 20 | Layer 16 |
|---|---|---|
| 100 | 74 → 61 | 62 → 52 |
| 500 | 89 → 86 | 83 → 84 |
| 1,000 | 89 → 85 | 83 → 79 |

With 100 observed queries, which episodes were seen matters a lot (±10–13). From 500, the two draws agree within 4.

**Summary:** across all four suites and two test sets, a linear map from layer 24 (replacing 8 Llama layers plus the head) stays close to the real model; layer 20 is suite- and episode-dependent. Deeper, a small MLP extends the reach to about layer 12–16.

Data: `$VLA_DATA/linear_followup/` (`mlp/`, `heads/`, `evals/`, `pairs/` for Goal/Object, `record/`).

---

## 13. Robustness under distribution shift

`split/run_robustness.sh`, `split/perturb.py` (2026-11-02). Each perturbation is applied to every episode of starts 40–49 and is deterministic per episode, so all three setups see identical perturbed scenes:
- **camera:** the third-person camera moved 6 cm sideways and 4 cm up; the wrist camera is unchanged.
- **objects:** every movable object shifted by up to 3 cm in x and y, redrawn if objects would overlap.
- **dark:** scene lights dimmed to 30% of their diffuse intensity (rendered; image ~25% darker).
- **noise:** Gaussian pixel noise, σ = 20/255, on both camera images.

Setups: the real model; the linear map in place of the head (§8 fold 4); Model 1 truncated after layer 20 plus the layer-20 linear map (§10–§11).

| Shift | Spatial: real / linear head / layer 20 | Long: real / linear head / layer 20 |
|---|---|---|
| none | 95 / 97 / 96 | 94 / 93 / 92 |
| camera | 97 / 99 / **83** | 83 / 82 / **68** |
| objects | 87 / 86 / 81 | 88 / 91 / **72** |
| dark | 98 / 97 / 94 | 91 / 89 / 84 |
| noise | 97 / 95 / 95 | 88 / 89 / 82 |

- **The linear head is as robust as the real model:** within −2 to +3 of it under every shift on both suites.
- **The layer-20 replacement is not:** it loses 14–15 points more than the real model when the camera moves, 6–16 when objects shift, and 2–7 under lighting and noise. The top Llama layers carry robustness that a linear readout from layer 20 does not reproduce, even where it matched the real model on the unperturbed scenes.

**Caveats:** one perturbation strength per kind, 100 episodes per cell, one seed; Spatial and Long only.

Data: `$VLA_DATA/robustness/evals/`.
