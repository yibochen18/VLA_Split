"""Open-loop diagnosis of black-box replacements (RESULTS §17): action error and hidden-state drift vs. the real model
on the 200 held-out report queries.

  cd openvla-oft && python ../split/diagnose_blackbox.py --suite libero_spatial --fits L1-9_tf2_n5000 ... --out x.json
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

LAYERS = (9, 16, 24, 32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--fits", nargs="+", required=True, help="Fit tags in $VLA_DATA/blackbox/fits/<suite>/")
    p.add_argument("--n", type=int, default=200, help="Report queries used (the last n of the shuffled order)")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def run(vla, head, q):
    out = vla.language_model(input_ids=None, attention_mask=q["mask"], inputs_embeds=q["emb"], use_cache=False,
                             output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    slots = slice(hs[-1].shape[1] - 57, hs[-1].shape[1] - 1)  # 56 action slots before the stop token
    act = head.normalized_actions(hs[-1][:, slots])
    return act, {layer: hs[layer][0].float() for layer in LAYERS}, slots


def main():
    args = parse_args()
    root = paths.DATA / "blackbox"
    record = root.parent / "secret_lora" / "record" / args.suite
    files = query_files(record)
    random.Random(0).shuffle(files)  # same order as fit_blackbox.py; its last 200 are the report queries
    files = files[-args.n:]

    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    vla = get_model(cfg).eval()
    head = Model2.load(paths.SPLIT / args.suite / "model2", "cuda")

    queries, real = [], []
    rec_err = []
    for f in files:
        d = torch.load(f, weights_only=False)
        q = {"emb": d["inputs_embeds"].cuda(), "mask": d["attention_mask"].cuda()}
        act, hs, slots = run(vla, head, q)
        rec = head.postprocess_gripper(head.unnormalize(act))
        rec_err.append(float(np.abs(rec - d["actions"]).max()))
        queries.append(q)
        real.append((act, {k: v.cpu() for k, v in hs.items()}, slots))
    print(f"real model vs recorded actions: max |diff| {max(rec_err):.2e}", flush=True)

    results = {"suite": args.suite, "n": len(files), "recorded_max_diff": max(rec_err), "fits": {}}
    for tag in args.fits:
        module, block, _ = blackbox.load(root / "fits" / args.suite / f"{tag}.pt", "cuda")
        hooks = blackbox.install_hooks(vla, module, block)
        act_err, grip, drift_slots, drift_all = [], [], {k: [] for k in LAYERS}, {k: [] for k in LAYERS}
        for q, (a0, h0, slots) in zip(queries, real):
            a, hs, _ = run(vla, head, q)
            act_err.append(float(np.abs(a - a0).mean()))
            grip.append(float((np.sign(a[:, -1] - 0.5) != np.sign(a0[:, -1] - 0.5)).mean()))
            for k in LAYERS:
                h, r = hs[k], h0[k].cuda()
                drift_slots[k].append(float((h[slots] - r[slots]).norm() / r[slots].norm()))
                drift_all[k].append(float((h - r).norm() / r.norm()))
        for h in hooks:
            h.remove()
        del module
        torch.cuda.empty_cache()
        r = {"action_l1": float(np.mean(act_err)), "action_l1_p90": float(np.percentile(act_err, 90)),
             "gripper_flip": float(np.mean(grip)),
             "drift_slots": {k: float(np.mean(v)) for k, v in drift_slots.items()},
             "drift_all": {k: float(np.mean(v)) for k, v in drift_all.items()}}
        results["fits"][tag] = r
        print(f"{tag:28s} action L1 {r['action_l1']:.4f} (p90 {r['action_l1_p90']:.4f}) gripper flips "
              f"{r['gripper_flip']:.3f} | drift on action slots " +
              " ".join(f"L{k} {r['drift_slots'][k]:.3f}" for k in LAYERS), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
