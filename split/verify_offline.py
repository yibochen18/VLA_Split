"""Verification rungs V1-V2: split vs unsplit on the saved reference dataset."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle
import paths
from model1 import Model1
from model2 import Model2
from split_client import SplitClient, start_server

CHECKPOINTS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}


# bound at import, before run_split_eval.py replaces it
_stock_process_action = rle.process_action


def expected_actions(query):
    """Reference final actions: the stock get_action output passed through process_action, row by row."""
    return np.stack([_stock_process_action(a, "openvla") for a in query["actions"]])


def report(name, diffs):
    ok = all(d == 0.0 for d in diffs)
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {sum(d == 0.0 for d in diffs)}/{len(diffs)} bitwise equal, max abs diff {max(diffs):.3g}")
    return {"pass": ok, "num_equal": int(sum(d == 0.0 for d in diffs)), "num": len(diffs), "max_abs_diff": max(diffs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--split_root", default=str(paths.SPLIT))
    p.add_argument("--socket", default="/tmp/vla_split_verify.sock")
    args = p.parse_args()

    queries = torch.load(args.reference, weights_only=False)["queries"]
    bundle = Path(args.split_root) / args.suite / "model2"
    results = {"reference": args.reference, "num_queries": len(queries)}

    # V1b first: cheap, and isolates Model 2 from Model 1
    model2 = Model2.load(bundle)
    v1b = [float(np.abs(model2(q["hidden"]) - expected_actions(q)).max()) for q in queries]
    results["V1b_model2_in_process"] = report("V1b Model 2 in-process", v1b)
    del model2
    torch.cuda.empty_cache()

    # V1a: Model 1 on the saved inputs
    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    model1 = Model1.load(cfg)
    print(f"Model 1 class: {type(model1.vla).__module__}.{type(model1.vla).__name__}; "
          f"norm_stats keys: { {k: list(v) for k, v in model1.vla.norm_stats.items()} }")
    v1a, hiddens = [], []
    for q in queries:
        obs = {"full_image": q["full_image"], "wrist_image": q["wrist_image"], "state": q["state"].copy()}
        hidden = model1(obs, q["task"]).cpu()
        hiddens.append(hidden)
        v1a.append(float((hidden.float() - q["hidden"].float()).abs().max()))
    results["V1a_model1"] = report("V1a Model 1", v1a)

    # V2: Model 2 in its own process, fed by Model 1's live outputs
    server = start_server(bundle, args.socket, Path(args.split_root) / f"server_verify_{args.suite}.log")
    try:
        client = SplitClient(args.socket)
        v2 = []
        for q, hidden in zip(queries, hiddens):
            actions, _ = client.query(hidden)
            v2.append(float(np.abs(actions - expected_actions(q)).max()))
        results["V2_two_processes"] = report("V2 two processes", v2)
        stats = client.stats()
        client.close()
    finally:
        server.terminate()
        server.wait()
    per_query = 32 + queries[0]["hidden"].numel() * 2
    results["server_stats"] = stats
    counters_ok = stats["num_queries"] == len(queries) and stats["bytes_in"] == len(queries) * per_query + 32
    print(f"Server counted {stats['num_queries']} queries, {stats['bytes_in']:,} bytes in "
          f"({'OK' if counters_ok else 'MISMATCH'}); openvla modules in server: {stats['loaded_modules_check'] or 'none'}")
    results["server_counters_ok"] = counters_ok

    out = Path(args.split_root) / args.suite / "verify_offline.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
