"""Model 1 time and memory per Llama depth (bitwise-checked), and Model 2 time on GPU and one CPU thread."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
import paths  # noqa: E402
from model1 import Model1  # noqa: E402
from model2 import Model2  # noqa: E402
from verify_offline import CHECKPOINTS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--depths", type=int, nargs="+", default=[32, 28, 24, 20, 16, 12, 8, 0],
                   help="Llama layers kept; 0 times everything except the Llama layers")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--start", type=int, default=40, help="Initial state used for the observations")
    p.add_argument("--mlp_bundle", default=None, help="default: $VLA_SPLIT/<suite>/model2")
    p.add_argument("--linear_bundle", default=None, help="default: $VLA_DATA/linear_head/heads/<suite>_fold4")
    p.add_argument("--out", required=True)
    return p.parse_args()


def timed(fn, reps, cuda, make_input=lambda: None):
    times = []
    for _ in range(reps):
        x = make_input()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if cuda:
            torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
    return times


def stats(times):
    return {"median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)), "p90_ms": float(np.percentile(times, 90)),
            "n": len(times)}


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu_others = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()

    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite, seed=7)
    set_seed_everywhere(cfg.seed)
    read = [d for d in args.depths if 0 < d < 32]
    m1 = Model1.load(cfg, layers=read)
    resize_size = get_image_resize_size(cfg)

    # Real observations, after the stabilization steps
    suite = benchmark.get_benchmark_dict()[args.suite]()
    obs = []
    for t in range(suite.n_tasks):
        env, desc = get_libero_env(suite.get_task(t), cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        o = env.set_init_state(suite.get_task_init_states(t)[args.start])
        for _ in range(cfg.num_steps_wait):
            o, *_ = env.step(get_libero_dummy_action(cfg.model_family))
        obs.append((rle.prepare_observation(o, resize_size)[0], desc))
        env.close()
    fresh = lambda i: (copy.deepcopy(obs[i][0]), obs[i][1])
    run = lambda x: m1(*x)

    # Reference tensors from full forward passes
    ref = []
    for i in range(len(obs)):
        cut_a, per_layer = run(fresh(i))
        ref.append({32: cut_a, **per_layer})
    m1.layers = ()  # return only the cut tensor

    llama = m1.vla.language_model.model
    layer_bytes = sum(p.numel() * p.element_size() for p in m1._all_layers[0].parameters())
    model1_bytes = sum(p.numel() * p.element_size() for p in m1.vla.parameters()) + \
        sum(p.numel() * p.element_size() for p in m1.proprio_projector.parameters())

    result = {"suite": args.suite, "observations": len(obs), "gpu_at_start": gpu_others, "rounds": args.rounds,
              "model1_weights_mb_full": model1_bytes / 2**20, "llama_layer_mb": layer_bytes / 2**20, "depths": {}}
    for d in args.depths:
        m1.set_depth(d)
        entry = {"llama_layers_kept": d,
                 "model1_weights_mb": (model1_bytes - (32 - d) * layer_bytes) / 2**20}
        if d > 0:
            entry["bitwise_equal_to_sweep"] = all(torch.equal(run(fresh(i)), r[d]) for i, r in enumerate(ref))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run(fresh(0))
        torch.cuda.synchronize()
        entry["activation_peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        result["depths"][d] = entry
        print(d, entry, flush=True)

    # Model 1 latency, depths interleaved
    rng = np.random.default_rng(0)
    times = {d: [] for d in args.depths}
    for d in args.depths:  # warmup
        m1.set_depth(d)
        timed(run, 3, cuda=True, make_input=lambda: fresh(0))
    for _ in range(args.rounds):
        for d in rng.permutation(args.depths):
            m1.set_depth(int(d))
            for i in range(len(obs)):
                times[int(d)] += timed(run, 1, cuda=True, make_input=lambda: fresh(i))
    for d in args.depths:
        result["depths"][d]["model1_latency"] = stats(times[d])
        print(d, result["depths"][d]["model1_latency"], flush=True)
    m1.set_depth(32)

    # Model 2 on the GPU and one CPU thread
    mlp_bundle = Path(args.mlp_bundle or paths.SPLIT / args.suite / "model2")
    linear_bundle = Path(args.linear_bundle or paths.DATA / "linear_head" / "heads" / f"{args.suite}_fold4")
    torch.set_num_threads(1)
    heads = {}
    for name, bundle in [("mlp", mlp_bundle), ("linear", linear_bundle)]:
        for device in ["cuda:0", "cpu"]:
            m2 = Model2.load(bundle, device)
            x = ref[0][32].to(device)
            m2(x)
            reps = 5 if (name, device) == ("mlp", "cpu") else 200
            heads[f"{name}_{device.split(':')[0]}"] = {
                **stats(timed(lambda _: m2(x), reps, cuda=device != "cpu")),
                "params": sum(p.numel() for p in m2.head.parameters()), "bundle": str(bundle)}
            print(name, device, heads[f"{name}_{device.split(':')[0]}"], flush=True)
            del m2
    result["model2"] = heads
    result["cpu_threads"] = 1
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
