# Results

Running record of measured results for the OpenVLA-OFT split project.

**Status (2026-11-02):** Baseline reproduced on all four suites (97.05% vs. 97.1%). Cut A split verified bitwise on all four suites, and full 500-episode split runs match the unsplit model exactly on every suite. Extraction tests (§6, §7): Model 2 can be copied from ~100 observed queries, even with up to 4 Llama layers moved into it. Linear-head study (§8): a single linear map distilled from observed traffic replaces Model 2 with no measurable loss in task success (97.15% vs. 96.85% for the MLP on identical scenes, 2,000 episodes), and is as robust as the real model under camera, lighting, noise and object-position shifts (§13). LIBERO's scenes depend on episode order (§9), so every comparison uses a real-model run done the same way. Depth (§10–§12): a linear map reading Llama layer 24 replaces the top 8 layers plus the head with small loss on all four suites (~15% faster); layer 20 (23% faster, 32% less memory) holds on Spatial but loses ~2–10 points on the other suites and is clearly less robust to camera and object shifts (§13). A small MLP readout recovers layers 12–16, so the fall-off there is the linear map's limit; at layers 4–8 the information is not yet there. Observation needed: ~100 queries for the head alone, ~2,500 for layer 20 on Spatial (§11). On a CPU, as in an edge TEE, one Llama layer takes 0.3–2 s per query, against ~400 ms per action chunk, so an edge TEE can hold a cut-A-sized readout but not transformer layers (§14). π0.5 (§15) reproduces at 96.9% vs. openpi's 96.85%, and one linear map replaces its ≈430M-parameter action expert with no significant loss overall (96.0% vs. 96.7% on identical scenes, 2,000 episodes; Object −2.2 pp), with queries 73% faster in uncompiled PyTorch; may be smaller in a compiled deployment. A small MLP readout does about as well as the linear map there (−1.1 pp vs. π0.5).

**Update (2026-11-12, LIBERO-Plus, §16):** on 10,030 perturbed LIBERO tasks (camera, robot start, language, light, background, sensor noise, object layout), with readouts fitted on clean LIBERO only. **OFT:** the linear map stays within ~1 pp of the real head (−1.1 pp, 10,030 paired tasks; at most −3.4 pp under any shift). **π0.5:** the readouts fall well behind the real expert (linear −14.8 pp, MLP −9.7 pp; camera −26 / −21, sensor noise −21 / −18), so π0.5's expert adds robustness that a readout of the pooled layer-18 input fitted on clean traffic does not reproduce, unlike OFT's head.

**Update (2026-11-15, secret beginning or middle, §17):** keeping the beginning or middle of OFT secret does not protect it either. OFT is the public `openvla-7b` + a rank-32 LoRA: plugging in the public weights already works for 8 secret layers on Spatial (90–98 vs. 95; Long 77–89 vs. 94), and for larger secret parts (layers 1–16, the vision encoder, vision + layers 1–8) a rank-32 LoRA fitted on 100 observed queries of the secret part's input and output brings Spatial back to 95–98 and Long to 86–90 (1,000 queries: 91–93; real 94). **Without public weights** (black box, 2026-11-16), a 2-layer transformer trained from scratch on observed traffic replaces layers 9–16 from 1,000 queries (Spatial 96) and reaches 92–99 at 5,000; layers 1–16 are harder (5,000 queries: Spatial 91, Long 80 vs. 94) but already mostly work. Unpublished weights raise the data the attacker needs by an order of magnitude or more; they do not stop the copy.

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

The later experiments each name their script in their section; README.md lists them in order.

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

The MLP side is the real MLP run exactly like each linear fold (alone, starting at start 10f), so both see identical scenes (§9); fold 0 comes from the full run, which also starts there. Folds 1–4 are reruns by `split/run_linear_refs.sh`, except Spatial fold 4, which is the determinism-check run of `split/run_linear_layers.sh` (`linear_layers/checks/`, 95/100). Paired comparison: exact McNemar test on the episodes where exactly one fails; 95% CI of the difference, normal approximation on the paired differences.

| Suite | Linear − MLP | 95% CI | McNemar p |
|---|---|---|---|
| Spatial | +0.6 pp | [−0.6, +1.8] | 0.51 |
| Object | −0.8 pp | [−2.0, +0.4] | 0.34 |
| Goal | +0.2 pp | [−1.3, +1.7] | 1.00 |
| Long | +1.2 pp | [−1.3, +3.7] | 0.44 |
| **All 2,000** | **+0.3 pp** | **[−0.6, +1.2]** | **0.57** |

History: the first version of this table paired the linear folds with the MLP's full 500-episode run (+0.1 pp, [−0.7, +0.9]). That pairing assumed identical scenes, which holds only for fold 0 (§9), so it was replaced by these same-scene runs. The MLP's own scores moved by up to 5 per 100 between the two kinds of run.

**Sanity checks:** the server reported 200,711 parameters in every closed-loop run (the linear map, not the MLP); each fitted head's training data excludes its test starts (train + test = all recorded queries); the linear map's actions equal the MLP's exactly in 0 of 1,330 held-out Spatial queries (it is a different policy, ~5% relative error on movement), and linear and MLP succeed on different episodes. Weak heads fail in the same harness (the gradient-trained linear copy from §6 on only 100 queries: 81/100; at deeper layers: 4–11/100).

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

Model 2 alone, per query (median of 200 calls; GPU in bf16, CPU in fp32, AMD Threadripper PRO 5955WX, one thread):

| | GPU | One CPU thread |
|---|---|---|
| MLP head (151M) | 0.49 ms | 60 ms |
| Linear map (0.2M) | 0.11 ms | 0.16 ms |

Replacing only the head saves almost nothing on the GPU (0.4 ms of ~100 ms); the saving comes from dropping Llama layers. On one CPU thread, a stand-in for a TEE core, the linear map is ~380× cheaper than the MLP head, though both fit easily within the ~400 ms an action chunk lasts (8 steps at LIBERO's 20 Hz).

*Correction (2026-11-02):* the first version of this table gave 537 ms for the MLP head on CPU. That run kept the head in bf16, which this CPU has no native support for (no AVX-512 BF16 or AMX), so PyTorch fell back to a slow path. Re-timed in fp32 (`bench_depth.py` now does this): 60 ms, with actions within 0.0025 of the bf16 head's. The earlier claims of "~3,400× faster" and "too slow for real-time control" were wrong.

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

---

## 14. What fits in a CPU TEE

`split/bench_tee.py` (2026-11-02). An edge TEE (Arm TrustZone, Intel SGX/TDX, AMD SEV-SNP) runs on the CPU, so whatever Model 2 contains has to run there. Time per query in fp32 on this server's CPU (AMD Threadripper PRO 5955WX; median of 5 calls for the layer, 100 for the heads). The Llama layer has random weights, since its time does not depend on them, and runs on OFT's real sequence: 606 tokens on Spatial (600–606 depending on the instruction: 513 vision and proprio tokens, the prompt, 56 action slots, the stop token).

| Part | Params | 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| One Llama-2-7B layer (606 tokens) | 202M | 2,053 ms | 700 ms | 322 ms |
| MLP head + post-processing | 151M | 59 ms | 23 ms | 18 ms |
| Linear map | 0.2M | 0.14 ms | 0.14 ms | 0.10 ms |

One query produces an 8-step action chunk, which lasts ~400 ms at LIBERO's 20 Hz. That is the most a query can take without the robot waiting, before counting Model 1's ~100 ms.

- **One Llama layer uses about the whole budget even on 8 desktop cores**, and this core already runs it near its peak (~120 GFLOP/s: one layer is ~250 GFLOP per query). An edge ARM core, e.g. Jetson Orin's Cortex-A78AE, has roughly a quarter of that peak fp32 throughput (from specifications, not measured), and TrustZone secure memory is often far smaller than one layer's 0.8 GB in fp32.
- **Heads fit easily:** the MLP head takes 18–59 ms, the linear map well under 1 ms.
- **Consequence for protection:** on an edge CPU TEE, Model 2 can only be a readout on top of Model 1's features, i.e. cut A or something of its size. §6 and §8 show such a readout is copied, or replaced by a linear map, from ~100 observed queries. Deeper cuts (§7, §10–§12) need transformer layers in the TEE, which do not fit there; they would need a GPU TEE (e.g. NVIDIA confidential computing), and a GPU TEE can hold the whole model without splitting it.

Data: `$VLA_DATA/linear_compute/bench/tee_spatial.json`.

---

## 15. π0.5: reproduction and linear replacement of the action expert

Repeats the §8 study on π0.5 (`pi05_libero`, openpi), whose action module is a ≈430M-parameter transformer (the "action expert": 311M in its transformer layers, 117M in the time conditioning (adaptive norms), 2M in the action projections; counted from the checkpoint, earlier written here as "300M") that builds each action chunk over 10 flow-matching steps from random noise, attending to the backbone's keys and values at every layer.

**Setup** (`split/pi05_server.py`, `split/run_pi05_eval.py`): openpi's PyTorch port, unmodified, in its own process and environment, run uncompiled. OFT's stock `run_episode` drives every episode (same env seed 0, wait steps, step limits, state and 180° image rotation); only the observation, the policy call and the action post-processing are π0.5's: images resized by openpi's own code on the server, the first 5 of 10 predicted actions executed, actions used as returned. Each query's noise comes from (seed 7, task, episode, query), so runs are reproducible from any starting episode.

**Checks:**

| Check | Result |
|---|---|
| Loads and answers | 3.6B params, 7.6 GB GPU, ~220 ms per query |
| Robot state unused (`pi05_libero` sets `discrete_state_input=False`) | Other or zero state → bitwise identical actions; other image → different actions |
| Determinism | Spatial, 1 episode per task, run twice in fresh processes: 236/236 queries bitwise identical, same successes |
| Recording changes nothing | Bitwise identical with recording on |

### Baseline (500 episodes per suite, seed 7)

| Suite | Ours | openpi reported | Difference | OFT (ours, §1) |
|---|---|---|---|---|
| Spatial | **99.4%** (497/500) | 98.8% | +0.6 | 98.6% |
| Object | **98.8%** (494/500) | 98.2% | +0.6 | 98.2% |
| Goal | **98.0%** (490/500) | 98.0% | 0.0 | 97.6% |
| Long | **91.4%** (457/500) | 92.4% | −1.0 | 93.8% |
| **Average** | **96.9%** | 96.85% | +0.05 | 97.05% |

All suites within ~1 pp of openpi's numbers, so the PyTorch port and our harness (env seed 0 instead of openpi's 7) reproduce π0.5. Long's weak spot is the same task as OFT's: "put both moka pots on the stove", 52% (OFT 72%), 24 of the 43 Long failures. These runs recorded every query's backbone states and actions (~4 MB per query; 27,976 queries on Long).

### Choosing the linear map's input (Spatial, offline)

`split/ridge_pi05.py`: closed-form ridge in dual form (queries × queries Gram matrix), standardized features, target = the 5 executed actions (35 numbers). λ chosen on starts 0–9 after fitting on 10–39, refit on 0–39, scored on held-out starts 40–49. Features = the 512 image tokens + the mean instruction token at one recorded layer, chosen before any closed-loop run by validation error (smaller input if within 5%).

| Input | Features | λ | Validation L1 | Held-out L1 | Train L1 | Gripper agreement |
|---|---|---|---|---|---|---|
| Layer-18 input (what the expert reads), all tokens | 1,050,624 | 10⁴ | **0.0381** | **0.0357** | 0.0034 | 98.7% |
| **Layer-18 input, 4 × 4 pooled** (chosen) | **67,584** | 10³ | 0.0388 | 0.0373 | 0.0133 | 98.7% |
| Final layer output (unused by the expert), all tokens | 1,050,624 | 10⁵ | 0.0415 | 0.0394 | 0.0106 | 98.7% |
| Always predict the mean action | | | | 0.198 | | |

**π0.5's own spread:** π0.5 samples its actions, so a deterministic map cannot match them exactly. Drawing 4 extra (unexecuted) samples per query on Spatial (1 episode per task, 236 queries): one sample differs from the mean of the other four by L1 **0.017** and from another single sample by 0.020; gripper signs agree 98.9%.

- The layer the action expert reads beats the final output, as at OFT's cut.
- Pooling 4 × 4 costs < 5% of error with 16× fewer features (2.4M-parameter map); the full input nearly memorizes its training set.
- The map's error is about twice π0.5's own sample spread; its gripper agreement is at π0.5's own level. Whether this suffices is the closed-loop test (next).

### Closed loop: the linear map in place of the action expert (5-fold, all suites)

`split/run_pi05_cv.sh`, `split/pi05_cv_summary.py`, as §8: for fold f, starts 10f–10f+9 of every task held out, the map fitted on the other 40 (λ on fold (f + 1) mod 5), and served in place of the action expert on the 100 held-out episodes. The server runs π0.5's own input transforms and backbone, stops at the input of the last backbone layer, and applies the map; the last layer and the expert never run. Checked: the map's first prediction of each episode online equals the offline prediction from the recording to 2e-6. π0.5's side is π0.5 run the same way (alone from start 10f, identical scenes, §9); fold 0 is the baseline run.

| Suite | Linear map | π0.5 (same scenes) | Episodes only linear fails / only π0.5 fails | Held-out L1 (mean of folds) |
|---|---|---|---|---|
| Spatial | **489/500 (97.8%)** | 492/500 (98.4%) | 10 / 7 | 0.037 |
| Object | **483/500 (96.6%)** | 494/500 (98.8%) | 17 / 6 | 0.030 |
| Goal | **490/500 (98.0%)** | 489/500 (97.8%) | 9 / 10 | 0.037 |
| Long | **457/500 (91.4%)** | 459/500 (91.8%) | 27 / 25 | 0.031 |
| **All** | **1919/2000 (96.0%)** | 1934/2000 (96.7%) | 63 / 48 | |

| Suite | Linear − π0.5 | 95% CI | McNemar p |
|---|---|---|---|
| Spatial | −0.6 pp | [−2.2, +1.0] | 0.63 |
| Object | −2.2 pp | [−4.1, −0.3] | 0.03 |
| Goal | +0.2 pp | [−1.5, +1.9] | 1.00 |
| Long | −0.4 pp | [−3.2, +2.4] | 0.89 |
| **All 2,000** | **−0.8 pp** | **[−1.8, +0.3]** | **0.18** |

**Splits and leakage (audited 2026-11-05, all 20 π0.5 and all 20 OFT §8 fits):** in every fold the test starts (10f–10f+9) and validation starts ((f + 1) mod 5) are disjoint, and train + test queries equal all recorded queries of the suite, so no test query is ever fitted. Standardization and λ use training data only; the final map is refit on train + validation and never sees test data. Observed queries per fit: π0.5 Spatial ~8,500, Object ~11,400, Goal ~9,300, Long ~22,400 (one row of 35 outputs per query); OFT §8 Spatial ~5,300, Object ~7,100, Goal ~5,800, Long ~13,600 (8 rows of 7 outputs per query, one map shared across the 8 steps). Same rule for both (every recorded query of the 40 non-test starts per task); the counts differ because π0.5 queries every 5 steps and OFT every 8. Held out are initial states, not tasks: the 10 tasks appear in training. One caveat: π0.5's input (layer-18 input, 4 × 4 pooled) was chosen once on validation starts 0–9, which are fold 0's test starts. The choice was between three options and went to the option that was *not* best on those starts (the 5% size rule picked pooled over all tokens), so it did not favor fold 0. Fold 0's linear scores are 100 / 98 / 100 / 90 (Spatial / Object / Goal / Long); Spatial's 100 is its highest fold (others 97–98), within noise, and dropping fold 0 entirely gives −0.6 pp over the other 1,600 episodes (−0.8 pp with it).

Object's loss is mostly two tasks: chocolate pudding (45 vs 50 of 50) and alphabet soup (44 vs 47); every other Object task is within one episode.

**Speed:** 59 ms per query alone (70 ms averaged over the CV runs, with a π0.5 server sharing the GPU) vs. 219 ms for π0.5: the map skips the last backbone layer and the expert's 10 denoising passes. Both measured in uncompiled PyTorch (compilation was turned off for run-to-run equality); the expert's 10 steps are many small operations, so much of its time may be per-operation overhead that compilation removes, and the saving may be smaller in a compiled deployment. Map: 2.4M parameters (67,584 × 35); π0.5: 3.6B.

**Takeaway:** one linear map, fitted in closed form from ~8,500–22,700 observed queries, (400 episodes per suite: 40 starts × 10 tasks, ~35 min (Spatial) to ~1.5 h (Long) of robot time at 20 Hz, the same episodes as OFT's §8 fits) replaces π0.5's ≈430M-parameter flow-matching action expert with no significant loss overall (−0.8 pp, CI [−1.8, +0.3], 2,000 paired episodes) and a faster query (73% faster in uncompiled PyTorch; may be smaller in a compiled deployment), the same picture as OFT's MLP head (§8, +0.3 pp). The exception is Object, a small but significant loss (−2.2 pp). Caveats: one seed; LIBERO only; how few queries suffice is not measured for π0.5 (OFT: ~100 for the head, §11); the expert's sampling is replaced by a deterministic map (π0.5's own samples differ by L1 ≈ 0.02, the map's error is ≈ 0.03–0.04).

**Run note:** Long's fits first ran out of GPU memory (22k × 22k matrices next to a π0.5 server); `ridge_pi05.py` now keeps those matrices in CPU memory (same Spatial fit: held-out L1 0.0373, weights within 4e-6), and only Long's linear folds were rerun.

Data: `$VLA_DATA/pi05/evals/` (runs), `$VLA_DATA/pi05/record/` (recordings, layer-18 input only after the choice), `$VLA_DATA/pi05/heads/features/` (input choice), `$VLA_DATA/pi05/cv/` (fold heads, closed-loop runs, `summary.json`).

### A small MLP in place of the action expert (5-fold, all suites)

`split/mlp_pi05.py`, `split/run_pi05_mlp_cv.sh`, `split/pi05_cv_summary.py mlp` (2026-11-06 to 11-08). Same input (layer-18 input, 4 × 4 pooled, 67,584 features), same folds, same targets and the same closed-loop protocol as the linear map above; only the map changes: per-feature standardization (training-set mean and spread, as the linear map) → 512 hidden units → ReLU → 35 outputs, 34.6M parameters. Trained with AdamW (lr 3e-4, weight decay 0.01, batch 64, cosine schedule), L1 loss on standardized targets; the number of steps is chosen on the validation starts from {3k, 6k, 12k, 24k, 48k} (as λ for the linear map), then the MLP is retrained on train + validation. Served by `pi05_server.py --mlp` on the same path as the linear map; online actions equal the offline predictions.

| Suite | MLP | π0.5 (same scenes) | MLP − π0.5 [95% CI], McNemar p | Linear − π0.5 (above) | Held-out L1: MLP / linear |
|---|---|---|---|---|---|
| Spatial | 498/500 | 492/500 | +1.2 [+0.1, +2.3], 0.07 | −0.6 | 0.032 / 0.037 |
| Object | 479/500 | 494/500 | −3.0 [−5.0, −1.0], 0.01 | −2.2 | 0.026 / 0.030 |
| Goal | 492/500 | 489/500 | +0.6 [−1.1, +2.3], 0.65 | +0.2 | 0.031 / 0.037 |
| Long | 443/500 | 459/500 | −3.2 [−6.2, −0.2], 0.05 | −0.4 | 0.025 / 0.031 |
| **All** | **1912/2000 (95.6%)** | 1934/2000 (96.7%) | **−1.1 [−2.1, −0.1], 0.05** | −0.8 | |

- **On clean LIBERO the MLP is no better than the linear map** (MLP − linear on the same episodes −0.4 pp [−1.4, +0.7], p = 0.59), although its held-out action error is ~17% lower: the linear map is already at π0.5's level there, and Object's small loss stays.
- **First recipe, superseded:** the first MLP used per-sample LayerNorm on its input (copied from §12, where it worked for OFT) and lost 7.8 pp to π0.5 (1779/2000; Long −23 pp). Not a serving bug, not leakage (same splits as the linear map), not gripper flicker. Cause: on π0.5's layer-18 input most of each feature is a fixed offset and the per-sample LayerNorm scale varies 70–200 across inputs, so every feature is rescaled by an input-dependent factor; the offline error was still lower than the linear map's, but closed loop it failed once the robot drifted off π0.5's trajectories. Switching to per-feature standardization fixed it (Long fold 0: 71 → 91 of 100; with dropout 0.2 and weight decay 0.1: 88). Note: this recipe choice was made on closed-loop success on Long fold 0, whose starts 0–9 are that fold's test starts, and the fold-0 heads are the ones used on LIBERO-Plus (§16); so the MLP recipe is partly selected on test results (the linear map's recipe is not). The conclusions do not rest on it: the MLP ends up no better than the linear map on clean LIBERO. Data: `$VLA_DATA/pi05/mlp_cv/` (LayerNorm), `$VLA_DATA/pi05/mlp_variants/`.

Data: `$VLA_DATA/pi05/mlp_cv_std/` (fold heads, closed-loop runs, `summary.json`); pooled features cached at `$VLA_DATA/pi05/features/pool4/`.

---

## 16. LIBERO-Plus: do the readouts hold up off-distribution?

**Status:** complete (2026-11-06 to 11-12): all 10,030 tasks × 5 setups, 50,150 episodes, no failed jobs.

**Question:** the readouts above were fitted and tested on clean LIBERO, a narrow benchmark. Do they only work because test scenes look like training scenes? LIBERO-Plus (arXiv 2510.13626, code at commit 4976dc3) perturbs the four LIBERO suites in seven ways: camera viewpoint, robot initial state, language (rewritten instructions), light, background texture, sensor noise (blur, noise and other image corruptions) and object layout; 10,030 tasks, one episode each, starting from LIBERO's initial state 0 (object-layout variants instead load LIBERO-Plus's own initial states from `libero_newobj/`).

**Setups** (`split/run_liberoplus.py`):
- `oft_real`, `pi05_real`: the LIBERO models as before (OFT per-suite checkpoints with their MLP heads through the cut-A split; `pi05_libero`). **Not retrained on LIBERO-Plus**, as on the LIBERO-Plus leaderboard.
- `oft_linear` (§8), `pi05_linear` (§15), `pi05_mlp` (§15, standardized MLP): the **fold-0 readouts fitted on clean LIBERO**. Fold 0 held out starts 0–9, so no readout saw start 0. Not refitted on LIBERO-Plus traffic: the question is whether a map fitted on normal operation holds up when conditions change.

**Pairing:** every task builds a fresh environment and runs one episode, so each readout and its real model see identical scenes (no §9 order effect). Checked per episode by the first camera image's MD5: identical in all 10,030 pairs except 20 (OFT linear, π0.5 MLP) and 28 (π0.5 linear), all Light tasks. Some LIBERO-Plus Light tasks render differently from run to run even alone on one GPU, so this is LIBERO-Plus's own nondeterminism, not our pairing. At most two simulators render per GPU: with up to seven, MuJoCo's offscreen rendering failed and camera images changed (tried and rejected 2026-11-07). Instructions identical on both sides of every pair (checked).

**Instructions** (`--plus_prompt`, `split/liberoplus_prompts.py`): LIBERO-Plus builds each task's instruction from its file name, so non-language variants get tags such as "… view 0 0 100 0 0 initstate 196" appended (LIBERO-Plus issues #64, #65, #71). We send the base task's original LIBERO instruction; language variants keep LIBERO-Plus's rewritten instruction. Consequence: our real-model numbers are higher than the leaderboard's (below) and not directly comparable to it.

### Success per suite (paired; difference in pp [95% CI], McNemar p)

| Suite | Tasks | OFT | OFT linear − OFT | π0.5 | π0.5 linear − π0.5 | π0.5 MLP − π0.5 |
|---|---|---|---|---|---|---|
| Spatial | 2,402 | 86.8% | −0.0 [−0.9, +0.9], 1.00 | 92.7% | −3.6 [−4.9, −2.3], <0.001 | −2.4 [−3.6, −1.2], <0.001 |
| Object | 2,518 | 69.9% | −1.2 [−2.2, −0.1], 0.04 | 91.3% | −17.3 [−18.9, −15.6], <0.001 | −13.6 [−15.2, −12.0], <0.001 |
| Goal | 2,591 | 71.6% | −1.3 [−2.3, −0.2], 0.02 | 81.5% | −18.7 [−20.5, −16.9], <0.001 | −9.3 [−10.9, −7.8], <0.001 |
| Long | 2,519 | 67.9% | −2.0 [−3.4, −0.6], 0.01 | 79.3% | −19.2 [−21.1, −17.3], <0.001 | −13.1 [−14.8, −11.3], <0.001 |
| **All** | **10,030** | **73.9%** | **−1.1 [−1.7, −0.6], <0.001** | **86.1%** | **−14.8 [−15.7, −14.0], <0.001** | **−9.7 [−10.5, −8.9], <0.001** |

### Per shift (all four suites)

| Shift | Tasks | OFT | OFT linear | π0.5 | π0.5 linear | π0.5 MLP |
|---|---|---|---|---|---|---|
| Camera | 1,599 | 64.4 | 63.6 | 76.5 | 51.0 | 55.6 |
| Robot initial state | 1,550 | 35.0 | 35.0 | 77.9 | 62.4 | 68.8 |
| Language | 1,537 | 83.3 | 81.5 | 86.3 | 73.9 | 81.2 |
| Light | 1,142 | 94.0 | 94.0 | 96.8 | 85.3 | 94.4 |
| Background | 1,076 | 95.1 | 94.2 | 96.2 | 88.0 | 90.4 |
| Sensor noise | 1,601 | 79.6 | 76.2 | 89.2 | 67.8 | 71.5 |
| Object layout | 1,525 | 77.9 | 77.4 | 85.8 | 80.1 | 83.0 |
| **All** | **10,030** | **73.9** | **72.8** | **86.1** | **71.2** | **76.4** |

| Shift | OFT linear − OFT | π0.5 linear − π0.5 | π0.5 MLP − π0.5 | π0.5 MLP − linear |
|---|---|---|---|---|
| Camera | −0.8 [−2.6, +0.9], 0.40 | −25.5 [−28.1, −23.0] | −20.9 [−23.3, −18.5] | +4.6 [+2.5, +6.7] |
| Robot initial state | +0.0 [−1.4, +1.4], 1.00 | −15.5 [−18.0, −13.1] | −9.2 [−11.4, −6.9] | +6.4 [+4.3, +8.5] |
| Language | −1.8 [−3.1, −0.5], 0.01 | −12.4 [−14.4, −10.5] | −5.1 [−6.8, −3.5] | +7.3 [+5.4, +9.2] |
| Light | +0.0 [−1.5, +1.5], 1.00 | −11.5 [−13.7, −9.2] | −2.4 [−3.9, −0.8], 0.003 | +9.1 [+6.9, +11.3] |
| Background | −0.8 [−2.3, +0.6], 0.31 | −8.2 [−10.2, −6.1] | −5.8 [−7.7, −3.8] | +2.4 [+0.3, +4.6], 0.03 |
| Sensor noise | −3.4 [−5.0, −1.9], <0.001 | −21.4 [−23.7, −19.2] | −17.7 [−19.9, −15.5] | +3.7 [+1.8, +5.7] |
| Object layout | −0.5 [−1.7, +0.6], 0.44 | −5.7 [−7.4, −4.0] | −2.9 [−4.4, −1.4] | +2.8 [+1.3, +4.4] |
| **All** | **−1.1 [−1.7, −0.6], <0.001** | **−14.8 [−15.7, −14.0]** | **−9.7 [−10.5, −8.9]** | **+5.2 [+4.4, +5.9]** |

π0.5 p-values not shown are < 0.001. Per suite and shift, readout − real (OFT linear / π0.5 linear / π0.5 MLP):

| Shift | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| Camera | −2.7 / −2.9 / −1.1 | −1.0 / **−41.9 / −35.1** | +1.7 / **−29.7 / −25.0** | −1.4 / **−26.3 / −21.2** |
| Robot initial state | +0.3 / −6.9 / −2.0 | −1.0 / −17.8 / −13.8 | +0.2 / −15.9 / −6.8 | +0.5 / −20.6 / −13.2 |
| Language | +1.0 / −2.1 / −1.8 | −2.3 / −5.6 / −5.1 | +0.2 / −23.2 / −3.2 | −6.5 / −17.8 / −10.7 |
| Light | +0.0 / +0.0 / −1.0 | +1.3 / −17.8 / −1.0 | −1.4 / −13.3 / +2.2 | +0.0 / −15.0 / −9.9 |
| Background | +1.2 / −1.6 / −0.4 | +0.4 / −6.5 / −4.8 | −4.3 / −13.9 / −7.1 | −0.3 / −10.0 / −10.0 |
| Sensor noise | +1.7 / −10.8 / −10.5 | −5.0 / −20.9 / −24.4 | −5.3 / −26.4 / −18.2 | −4.5 / −26.1 / −16.5 |
| Object layout | −1.3 / −0.3 / +0.5 | +0.7 / −5.2 / −3.2 | −1.4 / −6.6 / −3.8 | +0.0 / −11.9 / −5.4 |

### Comparison with the leaderboard

The LIBERO-Plus README reports OpenVLA-OFT (same per-suite checkpoints) at 69.6% overall (camera 56.4, robot 31.9, language 79.5, light 88.7, background 93.3, noise 75.8, layout 74.2). Ours is 73.9% (camera 64.4, robot 35.0, language 83.3, light 94.0, background 95.1, noise 79.6, layout 77.9): the same pattern (robot initial state by far the hardest, background and light easiest) and a few points higher in every category, as expected from sending clean instructions instead of tagged ones (issue #65 measured −8 pp for π0.5 on Object camera tasks from the tags). π0.5 is not on the leaderboard.

### Takeaways

- **OFT: the linear map holds up off-distribution.** A 200K-parameter map fitted on clean LIBERO stays within ~1 pp of OFT's 151M-parameter head over 10,030 perturbed tasks (−1.1 pp); the largest loss under any shift is sensor noise (−3.4 pp), and on Long's language variants (−6.5 pp). This extends §13 (four hand-made shifts, 100 episodes each) to LIBERO-Plus's seven kinds and all suites. OFT's head adds little that a linear readout of its cut-A input lacks, in or out of distribution.
- **π0.5: the readouts do not.** On clean LIBERO the linear map and the MLP were within ~1 pp of π0.5's expert; under LIBERO-Plus shifts they lose 14.8 and 9.7 pp, up to −42 pp (Object camera). π0.5 itself is much more robust than OFT here (86.1% vs. 73.9%; robot initial state 78% vs. 35%), and the readouts lose most of that advantage: π0.5 linear (71.2%) is below OFT. So the expert contributes robustness that a readout of the pooled layer-18 input, fitted on clean traffic, does not reproduce, unlike OFT's head.
- **The MLP recovers part of the gap** (+5.2 pp over linear; most of Light and Language, about half of Robot), but little of camera or sensor noise.
- **Open question: why.** π0.5's readouts see less than the expert: one layer, 4 × 4 pooled (the expert reads every layer's keys and values at full resolution over 10 denoising steps). Pooling was chosen on clean validation error (full tokens were only 1.8% better there), which says nothing about robustness. Next: a readout of the full-resolution layer-18 input on the worst cases (Object camera and sensor noise), to separate lost spatial detail from what the expert itself computes.

**Caveats:** one episode per task, one seed; readouts are a single fit (fold 0); real-model numbers use clean instructions (not comparable to the leaderboard one-to-one).

**Run cost:** ~3 days on two shared RTX 4090s (GPU 0 from 2026-11-09): one π0.5 job needs ~8 GB, one OFT job ~16 GB, at most two simulators rendering per GPU. Sensor-noise chunks ran 3–6× slower than others (image corruption on the CPU per frame).

Data: `$VLA_DATA/libero_plus/evals/<setup>/<suite>/` (one JSON per 250-task chunk, with per-episode success, first-image MD5 and instruction), `$VLA_DATA/libero_plus/summary.json` (`split/liberoplus_summary.py`).

---

## 17. Keeping the beginning or middle secret: can the attacker rebuild it?

**Status:** complete (2026-11-13 to 11-16): attack 1 (public weights), attack 2 (LoRA refit) and the black-box case (replacements trained from scratch), Spatial and Long.

**Question:** §6–8 and §10–12 showed that keeping the top of OFT secret (the head, or the head plus top layers) does not protect it. What if the secret part is at the **beginning** (vision encoder, projector, first Llama layers) or in the **middle** (e.g. Llama layers 9–16)? The attacker owns the robot and runs every other part, so they see the secret part's input and output on every query, and the actions; they do not see its weights or internal activations.

**Key fact:** OFT's LIBERO checkpoints are the public `openvla/openvla-7b` plus a merged rank-32 LoRA (openvla-oft `LIBERO.md`). Compared weight by weight (`split/public_diff.py`, Spatial checkpoint): each matrix we checked (all attention and MLP matrices of layers 1, 8, 16 and 32, the projector's first layer and one vision layer) is the public one plus a rank-32 change (96–99.9% of the change in its top 32 singular values). Relative change ||W_ft − W_pub|| / ||W_pub||: vision encoder 6%, projector 26%, Llama layers 10–17% (layers 1–16) rising to ~21% (layers 20–27); embeddings, final norm and `lm_head` unchanged. The proprio projector and the action head are new (no public version) and are never secret here.

### Attack 1: plug in the public weights (no data, no training)

`split/run_public_swap.sh`, `run_split_eval.py --public_swap`. The secret part is replaced by the corresponding `openvla-7b` weights; everything else is the fine-tuned model. Successes out of 100 on held-out starts 40–49; the real model scores 95 (Spatial) and 94 (Long) on the same scenes (§10).

| Secret part | Spatial | Long |
|---|---|---|
| Llama layers 9–16 | 90 | 77 |
| Llama layers 1–8 | 98 | 89 |
| Llama layers 1–16 | 65 | 12 |
| Vision encoder + projector | 72 | 6 |
| Vision + projector + layers 1–8 | 30 | 0 |
| All 32 Llama layers | 0 | 0 |
| Everything except head and proprio projector | 0 | 0 |

- **Eight secret layers give little protection on Spatial:** the public layers in their place score 90 (layers 9–16) and 98 (layers 1–8) vs. 95. On Long they cost more without any training: 77 and 89 vs. 94.
- **Larger secret parts stop this attack:** layers 1–16, the vision encoder + projector, or the vision encoder plus 8 layers drop Long to 0–12. The fine-tuning changes there matter for the task, so a zero-cost swap is not enough. Whether a small amount of training repairs them is attack 2.

**Caveats:** one seed, 100 episodes per cell (±~3–5); two suites.

### Attack 2: public weights + a LoRA fitted on recorded traffic

`split/run_secret_lora.sh`, `split/fit_secret_lora.py`. The attacker records N queries of the secret part's input and output (all tokens; `run_split_eval.py --record_full`; Spatial starts 0–39, 5,303 queries; Long starts 0–19, 6,713 queries; none from the test starts 40–49; the recording runs scored 395/400 and 188/200), then fits a rank-32 LoRA (α 16, as OFT's fine-tune) on the public weights of every linear layer in the secret part, minimizing the per-feature-scaled squared error between its output and the recorded output (AdamW lr 5e-4, 1,500 steps of 8 queries, cosine; same recipe for every N, no tuning). Check: the training forward pass with the fine-tuned weights reproduces the recorded outputs bitwise for all three secret parts. Secret parts: the three where attack 1 failed most (Long 0–12). Layers 9–16 and 1–8 were not refitted.

Successes out of 100, held-out starts 40–49 (real model 95 Spatial, 94 Long; attack 1 = public weights, no training):

| Secret part | Spatial: attack 1 | N = 100 | 1,000 | 5,000 | Long: attack 1 | N = 100 | 1,000 | 5,000 |
|---|---|---|---|---|---|---|---|---|
| Llama layers 1–16 | 65 | **95** | 96 | 97 | 12 | 86 | **92** | 91 |
| Vision encoder + projector | 72 | **98** | 97 | 97 | 6 | 90 | **93** | 95 |
| Vision + projector + layers 1–8 | 30 | **97** | 95 | 97 | 0 | 88 | **91** | 92 |

Error of the secret part's output on 200 held-out recorded queries (squared error per feature, scaled by the feature's spread; 1 ≈ no better than a constant):

| Secret part | Spatial: public → N = 100 → 5,000 | Long: public → N = 100 → 5,000 |
|---|---|---|
| Layers 1–16 | 0.27 → 0.024 → 0.012 | 0.40 → 0.046 → 0.026 |
| Vision + projector | 0.88 → 0.099 → 0.052 | 0.92 → 0.125 → 0.066 |
| Vision + projector + 1–8 | 0.41 → 0.039 → 0.020 | 0.54 → 0.066 → 0.035 |

Fit time on one RTX 4090: ~15 min (layers 1–16), ~40 min (vision), ~45 min (vision + layers 1–8).

### Takeaways

- **No secret part tested protects OFT.** With 100 observed queries (under a minute of robot time) and a 15–45 minute fit, every repaired part matches the real model on Spatial (95–98 vs. 95); on Long, 100 queries give 86–90 and 1,000 give 91–93 (real 94), within ~3 points.
- **The reason is the public starting point.** OFT is `openvla-7b` + a rank-32 change, so the attacker only has to recover a small low-rank correction, and the secret part's output gives them a rich target (every token's hidden state, not just 7 action numbers). Parts that the free swap broke (Long 0–12) are mostly repaired from 100 queries (Long 86–90) and to within ~3 points from 1,000.
- **Together with §6–12:** keeping the top (head or top layers), the middle (layers 9–16: public weights alone give 90 / 77) or the beginning (vision encoder, early layers) secret all fail, at ~100–1,000 observed queries. For a model fine-tuned from public weights, hiding part of it is not a protection on its own; what would have to stay secret is something not derivable from public weights plus a little observed traffic. Without public weights, see the black-box results below.

**Caveats:** one seed, 100 episodes per cell (±~3–5); one fit per cell with one recipe (an attacker could only do better); OFT only, Spatial and Long; attack 2 was run only on the three parts where attack 1 failed most, not on layers 9–16 or 1–8 or other blocks.

### Black box: no public weights, replacement trained from scratch

Attacks 1–2 rely on OFT's public base model. A proprietary edge model would not be published, so here the attacker has no related weights at all, as in the original extraction test (§6–7): they train a replacement for the secret block from scratch on its recorded input and output (all tokens), and plug it in between the parts they have. `split/run_blackbox.sh`, `split/fit_blackbox.py`, `split/blackbox.py`; same recordings as attack 2 (Spatial starts 0–39, Long 0–19; tests 40–49).

Replacements (all start as the identity, h_out = h_in, and learn the update):

| Replacement | Parameters | Training |
|---|---|---|
| Identity (block dropped) | 0 | none (control) |
| Linear map, per token, on standardized input | 16.8M | ridge, closed form |
| MLP, per token, 4096 → 8192 → 4096 | 67M | AdamW lr 1e-4 |
| Transformer, 2 Llama-style layers at OFT's width, random init | 405M | AdamW lr 3e-4 |
| Transformer, 4 layers | 810M | AdamW lr 3e-4 |

For comparison, the secret block is 1.6B parameters (layers 9–16) or 3.2B (layers 1–16). Loss: squared error per feature scaled by the target's spread. Up to 10,000 steps of 4 queries. **The attacker's budget is N queries in total:** 20% of them (at least 20) are held out to choose λ (then refit on all N) or the best step, the rest train. A separate 200 recorded queries, never used for any choice, give the error below. Served by hooks: the real block still runs and its output is replaced by the replacement's prediction (checked bitwise: layer 8 unchanged, layer 16 = the replacement's output).

Successes out of 100, held-out starts 40–49 (real model: Spatial 95, Long 94). N = 100 / 1,000 / 5,000 observed queries.

| Secret block | Replacement | Spatial | Long |
|---|---|---|---|
| Layers 9–16 | identity (dropped) | 12 | 0 |
| | linear | 40 / 52 / 63 | 47 / 69 / 74 |
| | MLP | 37 / 69 / 70 | 41 / 74 / 79 |
| | 2-layer transformer | 61 / **96** / **99** | 42 / 82 / **92** |
| | 4-layer transformer | 58 / **98** / **97** | 30 / 80 / **93** |
| Layers 1–16 | identity (dropped) | 0 | 0 |
| | linear | 0 / 0 / 0 | 0 / 0 / 0 |
| | MLP | 0 / 0 / 0 | 0 / 0 / 0 |
| | 2-layer transformer | 0 / 87 / **91** | 0 / 71 / 80 |
| | 4-layer transformer | 32 / 76 / 89 | 0 / 52 / 73 |

Error of the block's output on the 200 report queries (scaled squared error; identity = block dropped), N = 5,000:

| Secret block | Identity | Linear | MLP | 2-layer | 4-layer |
|---|---|---|---|---|---|
| 9–16, Spatial / Long | 0.50 / 0.58 | 0.18 / 0.27 | 0.15 / 0.23 | 0.083 / 0.14 | 0.074 / 0.13 |
| 1–16, Spatial / Long | 0.65 / 0.77 | 0.28 / 0.38 | 0.24 / 0.33 | 0.097 / 0.16 | 0.086 / 0.15 |

Fit time (N = 5,000, one RTX 4090): ~8 min (linear, MLP), ~27 min (2-layer), ~50 min (4-layer).

- **Even without public weights, a secret middle block is rebuilt.** A 2-layer transformer trained from scratch on 1,000 observed queries replaces layers 9–16 on Spatial (96 vs. 95); with 5,000, Spatial 99 and Long 92–93 (real 94). It has a quarter of the block's parameters. This beats the attacker's other option, skipping the block and reading actions from layer 8 (64–77 on Spatial, §12).
- **Layers 1–16 hold up better, but not fully.** With 5,000 queries a 2-layer transformer reaches 91 on Spatial and 80 on Long (−14), and the curve is still rising (Long 0 → 71 → 80). So 16 secret layers cost the attacker more data than 8, but at this budget the robot already works most of the time.
- **Per-token maps (linear, MLP) never work for layers 1–16** (0 everywhere) and only partly for 9–16 (63–79). OFT zeroes the input embeddings of the 56 action positions, so at layer 0 they are identical for every query: a per-token map gives them a constant, and their content has to come from other tokens through attention, which only the transformers have. By layer 8 the action positions already carry some of it. Dropping either block makes the robot fail (0–12).
- **Compared with the public-base case (same block, layers 1–16):** with public weights + LoRA, 100 queries give Spatial 95 and 1,000 give Long 92; from scratch, 5,000 queries give Spatial 91 and Long 80. Unpublished weights raise the data the attacker needs by well over an order of magnitude for this block, and at our largest budget the rebuild is still 4–14 points short, but it already works most of the time. For layers 9–16 (no LoRA run), public weights alone give 90 / 77 and a from-scratch rebuild reaches 96 / 82 at 1,000 queries and 99 / 92 at 5,000.

**Caveats:** one seed, 100 episodes per cell (±~3–5); one untuned recipe per replacement. The best training step was usually the last one (10,000), and λ for layers 1–16 sat at the edge of its grid, so longer training or more data would likely do better; these are lower bounds on what an attacker achieves. The vision encoder was not tested from scratch.

**Reference check (2026-11-16):** the real model rerun on starts 40–49 with the current code reproduces the references exactly (Spatial 95, Long 94; identical per-episode outcomes and query counts), so the code added for these attacks does not change the real model.

**Deployment note (§14):** one Llama layer takes ~0.3–2 s per query on a CPU against a ~400 ms budget, so a secret part of several Llama layers cannot run in an edge CPU TEE; it would need a GPU TEE or a remote server, where the attacker still sees the same input and output.

Data: `$VLA_DATA/public_swap/` (weight comparison `diff_<suite>.json`, evals), `$VLA_DATA/secret_lora/` (recordings, fits, evals), `$VLA_DATA/blackbox/` (fits with their logs and JSONs, evals).
