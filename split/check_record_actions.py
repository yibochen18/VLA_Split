"""Check that recorded executed actions match the real model replaying the recorded input (RESULTS §17); end-to-end
training uses them as targets.

  cd openvla-oft && python ../split/check_record_actions.py --suite libero_10 --n 1000 --out x.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle
from experiments.robot.robot_utils import get_model
import paths
from model2 import Model2
from secret_lora import query_files
from verify_offline import CHECKPOINTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    files = query_files(paths.DATA / "secret_lora" / "record" / args.suite)
    files = random.Random(0).sample(files, min(args.n, len(files)))
    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    vla = get_model(cfg).eval()
    head = Model2.load(paths.SPLIT / args.suite / "model2", "cuda")
    grip_q, grip_a, cont, raw_near = 0, 0, [], 0
    for f in files:
        d = torch.load(f, weights_only=False)
        with torch.no_grad():
            out = vla.language_model(input_ids=None, attention_mask=d["attention_mask"].cuda(),
                                     inputs_embeds=d["inputs_embeds"].cuda(), use_cache=False,
                                     output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        T = h.shape[1]
        norm = head.normalized_actions(h[:, T - 57:T - 1])
        act = head.postprocess_gripper(head.unnormalize(norm))
        rec = np.asarray(d["actions"])
        flips = int((np.sign(act[:, -1]) != np.sign(rec[:, -1])).sum())
        grip_a += flips
        grip_q += flips > 0
        raw_near += int((np.abs(norm[:, -1] - 0.5) < 0.05).sum())
        cont.append(float(np.abs(act[:, :6] - rec[:, :6]).max()))
    r = {"suite": args.suite, "n_queries": len(files), "gripper_mismatch_queries": grip_q,
         "gripper_mismatch_actions": grip_a, "actions_total": 8 * len(files),
         "gripper_raw_within_0.05_of_threshold": raw_near,
         "continuous_max_abs_diff": max(cont), "continuous_mean_of_per_query_max": float(np.mean(cont))}
    print(json.dumps(r, indent=2))
    Path(args.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
