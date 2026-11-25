"""FLOPs per query of π0.5, real model vs. the readouts that replace its action expert (as bench_flops.py).

Runs in openpi's environment:  vla_openpi python split/bench_flops_pi05.py --out ...
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_server import RESIZE, ReadoutPolicy, load_policy  # noqa: E402
from openpi_client import image_tools  # noqa: E402

PROMPTS = [  # one per suite, shortest to longest
    "open the middle drawer of the cabinet",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the alphabet soup and place it in the basket",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(os.environ.get("VLA_OPENPI", ""), "checkpoints/pi05_libero_pytorch"))
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--linear", default=os.path.join(os.environ.get("VLA_DATA", ""), "pi05/cv/heads/libero_spatial_fold0"))
    p.add_argument("--mlp", default=os.path.join(os.environ.get("VLA_DATA", ""), "pi05/mlp_cv_std/heads/libero_spatial_fold0"))
    p.add_argument("--out", required=True)
    return p.parse_args()


def element(prompt, rng):
    image = lambda: image_tools.convert_to_uint8(image_tools.resize_with_pad(
        rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), RESIZE, RESIZE))
    return {"observation/image": image(), "observation/wrist_image": image(),
            "observation/state": rng.standard_normal(8).astype(np.float32), "prompt": prompt}


def count(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    per_module = {name: int(sum(ops.values())) for name, ops in counter.get_flop_counts().items()}
    per_op = {str(op): int(n) for op, n in counter.get_flop_counts().get("Global", {}).items()}
    return int(counter.get_total_flops()), per_module, per_op


def measure(name, infer, result):
    rng = np.random.default_rng(0)
    totals, modules, ops = [], defaultdict(list), None
    for prompt in PROMPTS:
        total, per_module, ops = count(lambda: infer(element(prompt, rng)))
        totals.append(total)
        for k, v in per_module.items():
            modules[k].append(v)
    result["setups"][name] = {"flops_mean": float(np.mean(totals)), "flops_min": int(min(totals)),
                              "flops_max": int(max(totals)), "per_op_last": ops,
                              "per_module_mean": {k: float(np.mean(v)) for k, v in modules.items()}}
    print(f"{name}: {np.mean(totals) / 1e12:.4f} TFLOPs (min {min(totals) / 1e12:.4f}, max {max(totals) / 1e12:.4f})",
          flush=True)


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.checkpoint, args.config, args.device)
    model = policy._model
    result = {"torch": torch.__version__, "prompts": PROMPTS,
              "counted": "matmul/conv/attention FLOPs (2 per multiply-add); norms, activations, elementwise not counted",
              "params": sum(p.numel() for p in model.parameters()),
              "action_expert_params": sum(p.numel() for p in model.paligemma_with_expert.gemma_expert.parameters()),
              "num_flow_steps": getattr(model.config, "num_steps", None) if hasattr(model, "config") else None,
              "setups": {}}

    # The real model first: ReadoutPolicy installs a hook that stops the backbone before its last layer
    measure("real", lambda e: policy.infer(e), result)
    last_layer = model.paligemma_with_expert.paligemma.language_model.layers[-1]
    for kind, head_dir in [("pi05_linear", args.linear), ("pi05_mlp", args.mlp)]:
        last_layer._forward_pre_hooks.clear()
        readout = ReadoutPolicy(policy, head_dir, kind)
        measure(kind, readout.infer, result)
        result["setups"][kind]["readout_params"] = readout.params
        result["setups"][kind]["head_dir"] = head_dir
    last_layer._forward_pre_hooks.clear()

    full = result["setups"]["real"]["flops_mean"]
    for kind in ("pi05_linear", "pi05_mlp"):
        f = result["setups"][kind]["flops_mean"]
        result["setups"][kind].update(saved=full - f, saved_frac=(full - f) / full)
        print(f"{kind}: saves {(full - f) / 1e12:.4f} TFLOPs ({(full - f) / full:.2%})", flush=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
