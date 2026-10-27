"""Step 1 probe of the unsplit OpenVLA-OFT policy on LIBERO.

Runs a few episodes through the stock eval loop (openvla-oft is not modified) and:
  1. records a reference dataset of every model query: inputs, the 56x4096 hidden states
     that feed the action head (the cut-A interface), normalized and final actions;
  2. times each component per query (vision, projector, proprio, LLM, action head);
  3. repeats the same episodes with the same seed and checks the results are identical;
  4. re-runs the action head alone on the saved hidden states (a preview of the split);
  5. reports the action head's size and the interface size.

Usage (from openvla-oft/, so its `experiments` package is importable):
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python ../split/probe_baseline.py \
        --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --suite libero_spatial
"""

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

import experiments.robot.libero.run_libero_eval as rle
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes_per_task", type=int, default=1)
    p.add_argument("--passes", type=int, default=2, help="Repeat the same episodes to check determinism")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default="/mnt/ssd/yibo/reference")
    return p.parse_args()


class ComponentTimer:
    """Times module forwards with CUDA syncs, accumulating per model query."""

    def __init__(self):
        self.current = defaultdict(float)
        self._start = {}

    def attach(self, name, module):
        def pre(_mod, _inp):
            torch.cuda.synchronize()
            self._start[name] = time.perf_counter()

        def post(_mod, _inp, _out):
            torch.cuda.synchronize()
            self.current[name] += time.perf_counter() - self._start[name]

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)

    def pop(self):
        out, self.current = dict(self.current), defaultdict(float)
        return out


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def module_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = rle.GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.suite,
        num_trials_per_task=args.episodes_per_task,
        seed=args.seed,
    )
    rle.validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = rle.initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # --- Sizes -------------------------------------------------------------------------------
    sizes = {
        "action_head_params": count_params(action_head),
        "action_head_bytes": module_bytes(action_head),
        "action_head_dtype": str(next(action_head.parameters()).dtype),
        "proprio_projector_params": count_params(proprio_projector),
        "vla_params": count_params(model),
        "vla_bytes": module_bytes(model),
    }

    # --- Instrumentation -----------------------------------------------------------------------
    timer = ComponentTimer()
    timer.attach("vision_backbone", model.vision_backbone)
    timer.attach("projector", model.projector)
    timer.attach("proprio_projector", proprio_projector)
    timer.attach("language_model", model.language_model)
    timer.attach("action_head", action_head.model)

    # Capture the interface tensor: the head's input (hidden states) and output (normalized actions)
    captured = {}
    head_predict = action_head.predict_action

    def recording_predict_action(actions_hidden_states):
        normalized = head_predict(actions_hidden_states)
        captured["hidden"] = actions_hidden_states.detach().clone()
        captured["normalized"] = normalized.detach().float().cpu().clone()
        return normalized

    action_head.predict_action = recording_predict_action

    # Record every model query made by the stock run_episode loop
    queries = []
    episode_ctx = {}
    stock_get_action = rle.get_action

    def recording_get_action(cfg_, model_, obs, task_label, **kwargs):
        raw = copy.deepcopy(obs)  # get_vla_action normalizes obs["state"] in place
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        actions = stock_get_action(cfg_, model_, obs, task_label, **kwargs)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        queries.append({
            **episode_ctx,
            "query_idx": len([q for q in queries if q["episode_key"] == episode_ctx["episode_key"]]),
            "task": task_label,
            "full_image": raw["full_image"],
            "wrist_image": raw["wrist_image"],
            "state": raw["state"],
            "hidden": captured["hidden"].cpu(),
            "normalized_actions": captured["normalized"],
            "actions": np.stack(actions),
            "time_total": total,
            "times": timer.pop(),
        })
        return actions

    rle.get_action = recording_get_action

    # --- Run the same episodes `passes` times ------------------------------------------------
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    rle.check_unnorm_key(cfg, model)
    passes = []
    for pass_idx in range(args.passes):
        set_seed_everywhere(cfg.seed)
        queries = []
        successes = {}
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            for ep in range(cfg.num_trials_per_task):
                key = f"task{task_id}_ep{ep}"
                episode_ctx.clear()
                episode_ctx.update({"episode_key": key, "task_id": task_id, "episode": ep})
                success, _ = rle.run_episode(
                    cfg, env, task_description, model, resize_size, processor,
                    action_head, proprio_projector, noisy_action_projector, initial_states[ep],
                )
                successes[key] = bool(success)
                print(f"[pass {pass_idx}] {key} success={success} queries={sum(q['episode_key'] == key for q in queries)}")
            env.close()
        passes.append({"queries": queries, "successes": successes})

    ref = passes[0]["queries"]

    # --- Determinism across passes -----------------------------------------------------------
    determinism = None
    if len(passes) > 1:
        other = passes[1]["queries"]
        same_len = len(ref) == len(other)
        n = min(len(ref), len(other))
        hidden_equal = all(torch.equal(ref[i]["hidden"], other[i]["hidden"]) for i in range(n))
        action_equal = all(np.array_equal(ref[i]["actions"], other[i]["actions"]) for i in range(n))
        max_action_diff = max(float(np.abs(ref[i]["actions"] - other[i]["actions"]).max()) for i in range(n))
        determinism = {
            "num_queries": [len(ref), len(other)],
            "same_num_queries": same_len,
            "successes_equal": passes[0]["successes"] == passes[1]["successes"],
            "hidden_bitwise_equal": hidden_equal,
            "actions_bitwise_equal": action_equal,
            "max_abs_action_diff": max_action_diff,
        }

    # --- Head-only replay on saved hidden states (preview of the cut-A split) --------------------
    device = next(action_head.parameters()).device
    replay_diffs = []
    with torch.inference_mode():
        for q in ref:
            out = head_predict(q["hidden"].to(device)).float().cpu()
            replay_diffs.append(float((out - q["normalized_actions"]).abs().max()))
    head_replay = {
        "max_abs_diff": max(replay_diffs),
        "bitwise_equal": all(d == 0.0 for d in replay_diffs),
    }

    # --- Latency (first query excluded as warmup) ----------------------------------------------
    timed = ref[1:]
    latency_ms = {}
    for name in ["vision_backbone", "projector", "proprio_projector", "language_model", "action_head"]:
        vals = np.array([q["times"].get(name, 0.0) for q in timed]) * 1e3
        latency_ms[name] = {"mean": float(vals.mean()), "median": float(np.median(vals))}
    totals = np.array([q["time_total"] for q in timed]) * 1e3
    latency_ms["get_action_total"] = {"mean": float(totals.mean()), "median": float(np.median(totals))}
    component_sum = sum(latency_ms[k]["mean"] for k in latency_ms if k != "get_action_total")
    latency_ms["other_preprocessing_and_unnorm"] = {"mean": float(totals.mean() - component_sum)}

    hidden0 = ref[0]["hidden"]
    interface = {
        "shape": list(hidden0.shape),
        "dtype": str(hidden0.dtype),
        "bytes": hidden0.numel() * hidden0.element_size(),
    }

    summary = {
        "checkpoint": args.checkpoint,
        "suite": args.suite,
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "gpu": torch.cuda.get_device_name(0),
        "num_queries": len(ref),
        "successes": passes[0]["successes"],
        "success_rate": float(np.mean(list(passes[0]["successes"].values()))),
        "sizes": sizes,
        "interface": interface,
        "latency_ms": latency_ms,
        "determinism": determinism,
        "head_replay": head_replay,
    }

    stem = f"{args.suite}_seed{args.seed}_{args.episodes_per_task}ep"
    torch.save({"summary": summary, "queries": ref}, out_dir / f"{stem}.pt")
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved reference dataset to {out_dir / (stem + '.pt')}")


if __name__ == "__main__":
    main()
