"""Export a suite's action head and action stats as a standalone Model 2 bundle."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import paths  # before huggingface_hub, so HF_HOME is set
from huggingface_hub import hf_hub_download

# Same table as get_action_head in experiments/robot/openvla_utils.py
HEAD_FILES = {
    "libero_spatial": ("moojink/openvla-7b-oft-finetuned-libero-spatial", "action_head--150000_checkpoint.pt"),
    "libero_object": ("moojink/openvla-7b-oft-finetuned-libero-object", "action_head--150000_checkpoint.pt"),
    "libero_goal": ("moojink/openvla-7b-oft-finetuned-libero-goal", "action_head--50000_checkpoint.pt"),
    "libero_10": ("moojink/openvla-7b-oft-finetuned-libero-10", "action_head--150000_checkpoint.pt"),
}


def export(suite, out_root):
    repo_id, head_file = HEAD_FILES[suite]
    out_dir = Path(out_root) / suite / "model2"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(hf_hub_download(repo_id, head_file), map_location="cpu", weights_only=True)
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    torch.save(state_dict, out_dir / "head.pt")

    stats = json.loads(Path(hf_hub_download(repo_id, "dataset_statistics.json")).read_text())
    # Same key resolution as check_unnorm_key in run_libero_eval.py
    unnorm_key = suite if suite in stats else f"{suite}_no_noops"
    action_stats = {k: stats[unnorm_key]["action"][k] for k in ("q01", "q99", "mask")}
    (out_dir / "action_stats.json").write_text(json.dumps({"unnorm_key": unnorm_key, **action_stats}, indent=2))

    config = {
        "suite": suite,
        "source_checkpoint": repo_id,
        "source_head_file": head_file,
        "llm_dim": 4096,
        "action_dim": 7,
        "num_actions_chunk": 8,
        "dtype": "bfloat16",
        "normalization_type": "bounds_q99",
        "head_sha256": hashlib.sha256((out_dir / "head.pt").read_bytes()).hexdigest(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Exported {suite} -> {out_dir} ({sum(v.numel() for v in state_dict.values()):,} params)")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", nargs="+", default=["libero_spatial"])
    p.add_argument("--out_root", default=str(paths.SPLIT))
    args = p.parse_args()
    for s in args.suite:
        export(s, args.out_root)
