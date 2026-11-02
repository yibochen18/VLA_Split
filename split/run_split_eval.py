"""Closed-loop LIBERO evaluation with the split policy (Model 1 in-process, Model 2 over a socket)."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
import paths  # noqa: E402
import perturb  # noqa: E402
from model1 import Model1  # noqa: E402
from split_client import SplitClient, start_server  # noqa: E402
from verify_offline import CHECKPOINTS, expected_actions  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reference", default=None, help="probe_baseline.py output to compare against (V3)")
    p.add_argument("--split_root", default=str(paths.SPLIT))
    p.add_argument("--socket", default=None)
    p.add_argument("--bundle", default=None, help="Model 2 bundle to serve (default: <split_root>/<suite>/model2)")
    p.add_argument("--first_episode", type=int, default=0, help="First initial-state index per task")
    p.add_argument("--tag", default="", help="Suffix for output file names")
    p.add_argument("--save_pairs", default=None,
                   help="Directory to save every (interface tensor, actions) pair, one file per episode (extraction test data)")
    p.add_argument("--record_layers", type=int, nargs="*", default=[],
                   help="With --save_pairs: also save the action positions after these Llama layers to <save_pairs>/L<layer>/")
    p.add_argument("--interface_layer", type=int, default=None,
                   help="Send the action positions after this Llama layer to Model 2 instead of the cut A tensor "
                        "(layer-depth extraction test: Model 2 must then be a replacement trained on that layer)")
    p.add_argument("--truncate", action="store_true",
                   help="With --interface_layer: run only Llama layers 1..interface_layer instead of reading that "
                        "layer from a full forward pass (same tensor; measures the compute actually saved)")
    p.add_argument("--perturb", choices=perturb.KINDS, default="none",
                   help="Distribution shift applied to every episode (robustness test; see perturb.py)")
    return p.parse_args()


def compare_to_reference(queries, successes, reference_path):
    ref = torch.load(reference_path, weights_only=False)
    ref_queries, ref_successes = ref["queries"], ref["summary"]["successes"]
    by_episode, ref_by_episode = defaultdict(list), defaultdict(list)
    for q in queries:
        by_episode[q["episode_key"]].append(q)
    for q in ref_queries:
        ref_by_episode[q["episode_key"]].append(q)

    episodes_same_length = all(len(by_episode[k]) == len(ref_by_episode[k]) for k in ref_by_episode)
    hidden_diffs, action_diffs = [], []
    for key, ref_eps in ref_by_episode.items():
        for q, r in zip(by_episode[key], ref_eps):
            hidden_diffs.append(float((q["hidden"].float() - r["hidden"].float()).abs().max()))
            action_diffs.append(float(np.abs(q["actions"] - expected_actions(r)).max()))
    result = {
        "reference": reference_path,
        "episodes_compared": len(ref_by_episode),
        "same_queries_per_episode": episodes_same_length,
        "queries_compared": len(hidden_diffs),
        "hidden_bitwise_equal": all(d == 0.0 for d in hidden_diffs),
        "actions_bitwise_equal": all(d == 0.0 for d in action_diffs),
        "max_abs_hidden_diff": max(hidden_diffs),
        "max_abs_action_diff": max(action_diffs),
        "successes_equal": {k: successes.get(k) for k in ref_successes} == ref_successes,
    }
    result["pass"] = all(result[k] for k in ("same_queries_per_episode", "hidden_bitwise_equal", "actions_bitwise_equal", "successes_equal"))
    return result


def main():
    args = parse_args()
    out_dir = Path(args.split_root) / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"split_eval_seed{args.seed}_{args.episodes_per_task}ep"
    if args.first_episode:
        stem += f"_from{args.first_episode}"
    if args.tag:
        stem += f"_{args.tag}"
    bundle = Path(args.bundle) if args.bundle else out_dir / "model2"
    socket_path = args.socket or f"/tmp/vla_split_{args.suite}.sock"

    cfg = rle.GenerateConfig(
        pretrained_checkpoint=CHECKPOINTS[args.suite],
        task_suite_name=args.suite,
        num_trials_per_task=args.episodes_per_task,
        seed=args.seed,
    )
    set_seed_everywhere(cfg.seed)
    assert not args.record_layers or args.save_pairs, "--record_layers needs --save_pairs"
    assert not args.truncate or (args.interface_layer is not None and not args.record_layers), \
        "--truncate needs --interface_layer and no --record_layers"
    read_layer = args.interface_layer is not None and not args.truncate
    layers = sorted(set(args.record_layers) | ({args.interface_layer} if read_layer else set()))
    # Model 2 first, so a shared GPU can't fill up before it starts
    server = start_server(bundle, socket_path, out_dir / f"{stem}_server.log")
    try:
        model1 = Model1.load(cfg, layers=layers, depth=args.interface_layer if args.truncate else None)
    except BaseException:
        server.kill()
        raise
    model1_loaded_mb = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    resize_size = get_image_resize_size(cfg)
    client = SplitClient(socket_path)

    queries = []
    episode_ctx = {}
    keep_tensors = args.reference is not None
    episode_pairs = []
    episode_layers = {layer: [] for layer in args.record_layers}
    if args.save_pairs:
        Path(args.save_pairs).mkdir(parents=True, exist_ok=True)
        for layer in args.record_layers:
            (Path(args.save_pairs) / f"L{layer}").mkdir(exist_ok=True)

    def split_get_action(cfg_, model_, obs, task_label, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = model1(obs, task_label)
        per_layer = {}
        if layers:
            hidden, per_layer = hidden
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sent = per_layer[args.interface_layer] if read_layer else hidden
        actions, timings = client.query(sent)
        t2 = time.perf_counter()
        record = {**episode_ctx, "model1": t1 - t0, "total": t2 - t0, **timings}
        if keep_tensors:
            record.update(hidden=hidden.cpu(), actions=actions)
        if args.save_pairs:
            episode_pairs.append((hidden.cpu(), actions))
            for layer in args.record_layers:
                episode_layers[layer].append(per_layer[layer].cpu())
        queries.append(record)
        return list(actions)

    rle.get_action = split_get_action
    rle.process_action = lambda action, model_family: action
    if args.perturb == "noise":
        stock_prepare = rle.prepare_observation
        noise = {}

        def noisy_prepare_observation(obs, resize_size_):
            observation, img = stock_prepare(obs, resize_size_)
            if noise.get("episode") != episode_ctx["episode_key"]:
                noise.update(episode=episode_ctx["episode_key"], rng=perturb.episode_rng(episode_ctx["episode_key"]))
            for k in ("full_image", "wrist_image"):
                observation[k] = perturb.add_noise(observation[k], noise["rng"])
            return observation, img

        rle.prepare_observation = noisy_prepare_observation

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    successes = {}
    log_path = out_dir / f"{stem}.log"
    t_start = time.time()
    with open(log_path, "w") as log:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            perturb.perturb_env(env, args.perturb)
            for ep in range(args.first_episode, args.first_episode + cfg.num_trials_per_task):
                key = f"task{task_id}_ep{ep}"
                episode_ctx.clear()
                episode_ctx.update({"episode_key": key, "task_id": task_id, "episode": ep})
                success, _ = rle.run_episode(
                    cfg, env, task_description, model1.vla, resize_size, initial_state=initial_states[ep],
                )
                successes[key] = bool(success)
                if args.save_pairs:
                    actions_np = torch.from_numpy(np.stack([a for _, a in episode_pairs]))
                    meta = {"task_id": task_id, "episode": ep, "success": bool(success), "task": task_description}
                    torch.save({"hidden": torch.cat([h for h, _ in episode_pairs]), "actions": actions_np, **meta},
                               Path(args.save_pairs) / f"{key}.pt")
                    for layer, tensors in episode_layers.items():
                        torch.save({"hidden": torch.cat(tensors), "actions": actions_np, "layer": layer, **meta},
                                   Path(args.save_pairs) / f"L{layer}" / f"{key}.pt")
                        tensors.clear()
                    episode_pairs.clear()
                done = [successes[f"task{task_id}_ep{e}"] for e in range(args.first_episode, ep + 1)]
                line = (f"{key} success={success} | task {task_id}: {sum(done)}/{len(done)}"
                        f" | total: {sum(successes.values())}/{len(successes)} | {task_description}")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            env.close()

    server_stats = client.stats()
    client.close()
    server.terminate()
    server.wait()

    # Latency per query, first query excluded
    timed = queries[1:]
    latency_ms = {k: 1e3 * float(np.mean([q[k] for q in timed])) for k in ("model1", "serialize", "transfer", "server", "total")}
    latency_ms["server_breakdown"] = server_stats["mean_ms"]

    per_task = {}
    for task_id in range(task_suite.n_tasks):
        ks = [k for k in successes if k.startswith(f"task{task_id}_")]
        per_task[task_suite.get_task(task_id).language] = sum(successes[k] for k in ks) / len(ks)

    summary = {
        "suite": args.suite,
        "bundle": str(bundle),
        "interface_layer": args.interface_layer,
        "truncated": args.truncate,
        "perturb": args.perturb,
        "model1_gpu_mb": {"after_load": model1_loaded_mb, "peak": torch.cuda.max_memory_allocated() / 2**20},
        "first_episode": args.first_episode,
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "num_episodes": len(successes),
        "num_successes": sum(successes.values()),
        "success_rate": sum(successes.values()) / len(successes),
        "per_task_success": per_task,
        "num_queries": len(queries),
        "wall_clock_minutes": (time.time() - t_start) / 60,
        "latency_ms": latency_ms,
        "interface_bytes_per_query": queries[0]["request_bytes"],
        "server_stats": server_stats,
        "successes": successes,
    }
    if args.reference:
        summary["V3"] = compare_to_reference(queries, successes, args.reference)
        print(f"V3: {'PASS' if summary['V3']['pass'] else 'FAIL'} {json.dumps(summary['V3'], indent=2)}")
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"Success: {summary['num_successes']}/{summary['num_episodes']} ({100 * summary['success_rate']:.1f}%)")
    print(f"Latency ms: {json.dumps(latency_ms)}")
    print(f"Saved {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
