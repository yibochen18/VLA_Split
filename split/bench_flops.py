"""FLOPs per query of OpenVLA-OFT, real model vs. linear readouts at each depth (torch FlopCounterMode; matmuls,
convolutions and attention, 2 FLOPs per multiply-add)."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark
from torch.utils.flop_counter import FlopCounterMode

# torch 2.2's FlopCounterMode sees nothing that runs under inference_mode (get_vla_action, Model2); no_grad runs the
# same kernels and is counted
torch.inference_mode = lambda mode=True: torch.no_grad()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
import paths
from model1 import Model1
from model2 import Model2
from replacement import ReplacementModel2, forward_steps
from verify_offline import CHECKPOINTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--depths", type=int, nargs="+", default=[32, 28, 24, 20, 16, 12, 8])
    p.add_argument("--start", type=int, default=40, help="Initial state used for the observations")
    p.add_argument("--mlp_bundle", default=None, help="default: $VLA_SPLIT/<suite>/model2")
    p.add_argument("--linear_bundles", nargs="*", default=None,
                   help="LAYER=DIR; default: 32 = $VLA_DATA/linear_head/heads/<suite>_fold4, "
                        "24 = $VLA_DATA/linear_layers/heads/<suite>_L24")
    p.add_argument("--out", required=True)
    return p.parse_args()


def count(fn):
    """Returns (total FLOPs, {module: FLOPs}, {op: FLOPs}) of one call."""
    with FlopCounterMode(display=False) as counter:
        fn()
    per_module = {name: int(sum(ops.values())) for name, ops in counter.get_flop_counts().items()}
    per_op = {str(op): int(n) for op, n in counter.get_flop_counts().get("Global", {}).items()}
    return int(counter.get_total_flops()), per_module, per_op


def observations(cfg, suite_name, start):
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    obs = []
    for t in range(suite.n_tasks):
        env, desc = get_libero_env(suite.get_task(t), cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        o = env.set_init_state(suite.get_task_init_states(t)[start])
        for _ in range(cfg.num_steps_wait):
            o, *_ = env.step(get_libero_dummy_action(cfg.model_family))
        obs.append((rle.prepare_observation(o, resize_size)[0], desc))
        env.close()
    return obs


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite, seed=7)
    set_seed_everywhere(cfg.seed)
    m1 = Model1.load(cfg)
    obs = observations(cfg, args.suite, args.start)
    # get_vla_action normalizes obs["state"] in place: copy before every call (see bench_depth.py)
    fresh = lambda i: (copy.deepcopy(obs[i][0]), obs[i][1])

    model1_params = sum(p.numel() for p in m1.vla.parameters()) + sum(p.numel() for p in m1.proprio_projector.parameters())
    layer_params = sum(p.numel() for p in m1._all_layers[0].parameters())
    result = {"suite": args.suite, "observations": len(obs), "torch": torch.__version__,
              "counted": "matmul/conv/attention FLOPs (2 per multiply-add); norms, activations, elementwise not counted",
              "model1_params_full": model1_params, "llama_layer_params": layer_params, "model1": {}, "heads": {}}

    hidden = None
    for d in args.depths:
        m1.set_depth(d)
        totals, modules, ops = [], defaultdict(list), None
        for i in range(len(obs)):
            box = {}
            total, per_module, ops = count(lambda: box.__setitem__("h", m1(*fresh(i))))
            totals.append(total)
            for k, v in per_module.items():
                modules[k].append(v)
            if d == 32 and i == 0:
                hidden = box["h"]
        result["model1"][d] = {
            "llama_layers_kept": d, "params": model1_params - (32 - d) * layer_params,
            "flops_mean": float(np.mean(totals)), "flops_min": int(min(totals)), "flops_max": int(max(totals)),
            "per_module_mean": {k: float(np.mean(v)) for k, v in modules.items()}, "per_op_last": ops}
        print(f"depth {d}: {np.mean(totals) / 1e12:.4f} TFLOPs (min {min(totals) / 1e12:.4f}, max {max(totals) / 1e12:.4f})",
              flush=True)
    m1.set_depth(32)

    # Heads: the interface tensor has the same shape at every layer, so one input does for all
    mlp_bundle = Path(args.mlp_bundle or paths.SPLIT / args.suite / "model2")
    linear = dict(x.split("=", 1) for x in args.linear_bundles) if args.linear_bundles else {
        "32": paths.DATA / "linear_head" / "heads" / f"{args.suite}_fold4",
        "24": paths.DATA / "linear_layers" / "heads" / f"{args.suite}_L24"}
    m2 = Model2.load(mlp_bundle, "cuda:0")
    total, _, ops = count(lambda: m2.normalized_actions(hidden))
    result["heads"]["mlp"] = {"flops": total, "per_op": ops, "params": sum(p.numel() for p in m2.head.parameters()),
                              "bundle": str(mlp_bundle)}
    print(f"mlp head: {total / 1e9:.4f} GFLOPs", flush=True)
    for layer, bundle in linear.items():
        r = ReplacementModel2.load(bundle, "cuda:0")
        with torch.no_grad():
            total, _, ops = count(lambda: forward_steps(r.head, hidden.to("cuda:0")))
        result["heads"][f"linear_L{layer}"] = {"flops": total, "per_op": ops, "bundle": str(bundle),
                                                "params": sum(p.numel() for p in r.head.parameters())}
        print(f"linear L{layer}: {total / 1e6:.4f} MFLOPs", flush=True)

    full = result["model1"][32]["flops_mean"] + result["heads"]["mlp"]["flops"]
    result["comparisons"] = {"real": {"flops": full}}
    for layer in linear:
        f = result["model1"][int(layer)]["flops_mean"] + result["heads"][f"linear_L{layer}"]["flops"]
        result["comparisons"][f"linear_L{layer}"] = {"flops": f, "saved": full - f, "saved_frac": (full - f) / full}
    for k, v in result["comparisons"].items():
        print(k, {kk: (f"{vv:.4%}" if kk == "saved_frac" else f"{vv / 1e12:.4f} T") for kk, vv in v.items()}, flush=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
