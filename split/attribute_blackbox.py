"""Which tokens' errors break a black-box replacement (RESULTS §17): splice its output into the real block's one
token group at a time and measure the action error.

  cd openvla-oft && python ../split/attribute_blackbox.py --suite libero_spatial --fits L1-9_tf2_n5000 --out x.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.robot_utils import get_model  # noqa: E402
import paths  # noqa: E402,F401
import blackbox  # noqa: E402
from model2 import Model2  # noqa: E402
from secret_lora import query_files  # noqa: E402
from verify_offline import CHECKPOINTS  # noqa: E402

HUGE = 300.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    return p.parse_args()


def groups(h_real):
    """Token groups (boolean masks) from the real block output (T, D)."""
    T = h_real.shape[0]
    idx = torch.arange(T, device=h_real.device)
    huge = h_real.float().norm(dim=1) > HUGE
    prompt = (idx >= 514) & (idx < T - 57)
    g = {"bos": idx == 0, "patches": (idx >= 1) & (idx < 513), "proprio": idx == 513,
         "prompt_huge": prompt & huge, "prompt_other": prompt & ~huge, "action_slots": (idx >= T - 57) & (idx < T - 1),
         "stop": idx == T - 1}
    other_huge = huge & ~prompt
    if other_huge.any():
        g["huge_elsewhere"] = other_huge
    return g


def main():
    args = parse_args()
    root = paths.DATA / "blackbox"
    files = query_files(paths.DATA / "secret_lora" / "record" / args.suite)
    random.Random(0).shuffle(files)
    files = files[-args.n:]
    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    vla = get_model(cfg).eval()
    head = Model2.load(paths.SPLIT / args.suite / "model2", "cuda")
    layers = vla.language_model.model.layers
    state = {"mode": None}

    def actions(q):
        with torch.no_grad():
            out = vla.language_model(input_ids=None, attention_mask=q["mask"], inputs_embeds=q["emb"],
                                     use_cache=False, output_hidden_states=True, return_dict=True)
        T = out.hidden_states[-1].shape[1]
        return head.normalized_actions(out.hidden_states[-1][:, T - 57:T - 1])

    results = {"suite": args.suite, "n": len(files), "fits": {}}
    queries = [{"emb": d["inputs_embeds"].cuda(), "mask": d["attention_mask"].cuda()}
               for d in (torch.load(f, weights_only=False) for f in files)]
    real = [actions(q) for q in queries]
    sizes = {}

    for tag in args.fits:
        module, (a, b), _ = blackbox.load(root / "fits" / args.suite / f"{tag}.pt", "cuda")
        held = {}

        def keep_input(mod, a_, kw):
            held["h"] = a_[0] if a_ else kw["hidden_states"]

        @torch.no_grad()
        def splice(mod, a_, kw, output):
            h_real = output[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h_rep = module(held.pop("h")).to(h_real.dtype)
            g = groups(h_real[0])
            for k, m in g.items():
                sizes.setdefault(k, []).append(int(m.sum()))
            mode, name = state["mode"]
            if mode == "full":
                use = torch.ones_like(g["bos"])
            elif mode == "only":
                use = g[name]
            else:
                use = ~g[name]
            h = torch.where(use[None, :, None], h_rep, h_real)
            return (h,) + tuple(output[1:])

        hooks = [layers[a - 1].register_forward_pre_hook(keep_input, with_kwargs=True),
                 layers[b - 1].register_forward_hook(splice, with_kwargs=True)]
        names = ["bos", "patches", "proprio", "prompt_huge", "prompt_other", "action_slots", "stop"]
        res = {}
        for mode, name in [("full", None)] + [(m, n) for m in ("only", "all_but") for n in names]:
            state["mode"] = (mode, name)
            err = [float(np.abs(actions(q) - r).mean()) for q, r in zip(queries, real)]
            res[mode if name is None else f"{mode}:{name}"] = float(np.mean(err))
        for h in hooks:
            h.remove()
        results["fits"][tag] = res
        print(f"\n{tag}: full replacement {res['full']:.4f}")
        for name in names:
            print(f"  {name:13s} only {res['only:' + name]:.4f}   all but {res['all_but:' + name]:.4f}")
        Path(args.out).write_text(json.dumps(results, indent=2))
    print("\ntokens per group (mean):", {k: round(sum(v) / len(v), 1) for k, v in sizes.items()})


if __name__ == "__main__":
    main()
