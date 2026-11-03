"""Closed-loop LIBERO evaluation of π0.5, using the same episode loop as the OpenVLA-OFT runs (run_split_eval.py).

openvla-oft's stock run_episode drives the episode (same env seed, wait steps, step limits and success check); only the
observation, the policy call and the action post-processing are swapped for π0.5's: images rotated 180° and sent raw
(the server applies openpi's resize), the first 5 of the 10 predicted actions executed, actions used as returned.
π0.5 runs in its own process and environment (pi05_server.py). Run from openvla-oft/:
    python ../split/run_pi05_eval.py --suite libero_spatial --episodes_per_task 1
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle  # noqa: E402
from experiments.robot.robot_utils import set_seed_everywhere  # noqa: E402
import paths  # noqa: E402
from pi05_protocol import MAGIC_END, MAGIC_STATS, actions_from_bytes, encode_request, read_response  # noqa: E402

REPLAN_STEPS = 5  # actions executed per query, as in openpi's LIBERO client
SPLIT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--first_episode", type=int, default=0, help="First initial-state index per task")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise_seed", type=int, default=7, help="Base seed of π0.5's per-query noise")
    p.add_argument("--out_root", default=str(paths.DATA / "pi05" / "evals"))
    p.add_argument("--tag", default="", help="Suffix for output file names")
    p.add_argument("--socket", default=None)
    p.add_argument("--record_dir", default=None, help="Server saves the backbone states of every query here")
    p.add_argument("--noise_samples", type=int, default=0,
                   help="Server also draws this many unexecuted samples per query (saved to <stem>_samples/)")
    p.add_argument("--save_actions", action="store_true", help="Save every query's 10 x 7 actions (<stem>_actions.npz)")
    p.add_argument("--compare_to", default=None, help="Another run's _actions.npz: check the actions are bitwise equal")
    return p.parse_args()


def start_server(args, socket_path, log_path, samples_dir=None):
    """Launch pi05_server.py in openpi's environment and wait until it is listening."""
    cmd = f'source "{paths.REPO / "env.sh"}" && vla_openpi python "{SPLIT_DIR / "pi05_server.py"}" ' \
          f'--socket "{socket_path}" --seed {args.noise_seed}' + (f' --record_dir "{args.record_dir}"' if args.record_dir else "") \
          + (f' --noise_samples {args.noise_samples} --samples_dir "{samples_dir}"' if args.noise_samples else "")
    log = open(log_path, "w")
    # Own process group: bash -> uv -> python, so stop_server can end all of them
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
    for line in proc.stdout:
        if line.startswith("READY"):
            return proc
    stop_server(proc)
    raise RuntimeError(f"π0.5 server failed to start; see {log_path}")


def stop_server(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait()


class Pi05Client:
    def __init__(self, socket_path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)
        self.next_id = 0

    def _call(self, request):
        t0 = time.perf_counter()
        self.sock.sendall(request)
        request_id, status, payload, server_seconds, _ = read_response(self.sock)
        assert request_id == self.next_id and status == 0, (request_id, self.next_id, status)
        self.next_id += 1
        return payload, time.perf_counter() - t0, server_seconds

    def query(self, meta, image, wrist_image, state):
        """Returns (actions (10, 7) float64, timings in seconds)."""
        payload, total, server = self._call(encode_request(self.next_id, meta, image, wrist_image, state))
        return actions_from_bytes(payload), {"total": total, "server": server}

    def end_episode(self, meta):
        self._call(encode_request(self.next_id, meta, magic=MAGIC_END))

    def stats(self):
        return json.loads(self._call(encode_request(self.next_id, {}, magic=MAGIC_STATS))[0])

    def close(self):
        self.sock.close()


def compare_actions(actions, reference_path):
    ref = np.load(reference_path)
    common = sorted(set(actions) & set(ref.files))
    same_length = all(actions[k].shape == ref[k].shape for k in common)
    diffs = [float(np.abs(actions[k] - ref[k]).max()) for k in common if actions[k].shape == ref[k].shape]
    return {"reference": reference_path, "episodes_compared": len(common), "same_queries_per_episode": same_length,
            "actions_bitwise_equal": same_length and all(d == 0.0 for d in diffs),
            "max_abs_action_diff": max(diffs) if diffs else None}


def main():
    args = parse_args()
    out_dir = Path(args.out_root) / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pi05_eval_seed{args.seed}_{args.episodes_per_task}ep"
    if args.first_episode:
        stem += f"_from{args.first_episode}"
    if args.tag:
        stem += f"_{args.tag}"
    socket_path = args.socket or f"/tmp/vla_pi05_{args.suite}.sock"

    cfg = rle.GenerateConfig(task_suite_name=args.suite, num_trials_per_task=args.episodes_per_task, seed=args.seed,
                             num_open_loop_steps=REPLAN_STEPS)
    set_seed_everywhere(cfg.seed)
    server = start_server(args, socket_path, out_dir / f"{stem}_server.log", out_dir / f"{stem}_samples")
    try:
        main_loop(args, cfg, stem, out_dir, socket_path)
    finally:
        stop_server(server)


def main_loop(args, cfg, stem, out_dir, socket_path):
    client = Pi05Client(socket_path)

    queries, actions_by_episode, episode_ctx = [], {}, {}

    def pi05_prepare_observation(obs, resize_size):
        img = np.ascontiguousarray(get_libero_image(obs))
        observation = {
            "image": img,
            "wrist_image": np.ascontiguousarray(get_libero_wrist_image(obs)),
            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
        }
        return observation, img

    def pi05_get_action(cfg_, model_, observation, task_label, **kwargs):
        meta = {**episode_ctx, "query": episode_ctx["queries"], "prompt": str(task_label)}
        del meta["queries"], meta["episode_key"]
        actions, timings = client.query(meta, observation["image"], observation["wrist_image"], observation["state"])
        episode_ctx["queries"] += 1
        queries.append({"episode_key": episode_ctx["episode_key"], **timings})
        actions_by_episode.setdefault(episode_ctx["episode_key"], []).append(actions)
        return list(actions[:REPLAN_STEPS])

    rle.prepare_observation = pi05_prepare_observation
    rle.get_action = pi05_get_action
    rle.process_action = lambda action, model_family: action
    rle.NUM_ACTIONS_CHUNK = REPLAN_STEPS  # only silences run_episode's chunk-size warning

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    successes = {}
    t_start = time.time()
    with open(out_dir / f"{stem}.log", "w") as log:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            for ep in range(args.first_episode, args.first_episode + cfg.num_trials_per_task):
                key = f"task{task_id}_ep{ep}"
                episode_ctx.clear()
                episode_ctx.update({"episode_key": key, "task_id": task_id, "episode": ep, "queries": 0})
                success, _ = rle.run_episode(cfg, env, task_description, None, None, initial_state=initial_states[ep])
                successes[key] = bool(success)
                client.end_episode({"episode_key": key, "task_id": task_id, "episode": ep, "success": bool(success),
                                    "task": task_description})
                done = [successes[f"task{task_id}_ep{e}"] for e in range(args.first_episode, ep + 1)]
                line = (f"{key} success={success} | task {task_id}: {sum(done)}/{len(done)}"
                        f" | total: {sum(successes.values())}/{len(successes)} | {task_description}")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            env.close()

    server_stats = client.stats()
    client.close()

    actions = {k: np.stack(v) for k, v in actions_by_episode.items()}
    if args.save_actions:
        np.savez(out_dir / f"{stem}_actions.npz", **actions)
    per_task = {}
    for task_id in range(task_suite.n_tasks):
        ks = [k for k in successes if k.startswith(f"task{task_id}_")]
        per_task[task_suite.get_task(task_id).language] = sum(successes[k] for k in ks) / len(ks)
    timed = queries[1:]  # first query excluded (warmup)
    summary = {
        "suite": args.suite,
        "policy": "pi05_libero (openpi PyTorch, uncompiled)",
        "first_episode": args.first_episode,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "episodes_per_task": args.episodes_per_task,
        "num_episodes": len(successes),
        "num_successes": sum(successes.values()),
        "success_rate": sum(successes.values()) / len(successes),
        "per_task_success": per_task,
        "num_queries": len(queries),
        "wall_clock_minutes": (time.time() - t_start) / 60,
        "latency_ms": {k: 1e3 * float(np.mean([q[k] for q in timed])) for k in ("total", "server")},
        "record_dir": args.record_dir,
        "server_stats": server_stats,
        "successes": successes,
    }
    if args.compare_to:
        summary["compare"] = compare_actions(actions, args.compare_to)
        print(f"Compare: {json.dumps(summary['compare'])}")
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"Success: {summary['num_successes']}/{summary['num_episodes']} ({100 * summary['success_rate']:.1f}%)")
    print(f"Latency ms: {json.dumps(summary['latency_ms'])}")
    print(f"Saved {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
