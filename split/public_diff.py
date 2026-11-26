"""How far each part of an OFT checkpoint moved from the public openvla-7b it was LoRA-fine-tuned from.

Per part (vision, projector, each Llama layer, embeddings): relative change ||W_ft - W_pub|| / ||W_pub||, and for
the attention/MLP matrices of a few layers the rank of the change (a merged rank-32 LoRA gives ~32 large
singular values, then bf16 rounding noise)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

import paths  # sets HF_HOME
from verify_offline import CHECKPOINTS


def tensors(repo):
    root = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    return root, weight_map


def part_of(key):
    if key.startswith("vision_backbone."):
        return "vision"
    if key.startswith("projector."):
        return "projector"
    if key.startswith("language_model.model.layers."):
        return f"llm{int(key.split('.')[3]) + 1:02d}"
    return key.rsplit(".", 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--rank_layers", type=int, nargs="*", default=[1, 8, 16, 32])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pub_root, pub_map = tensors("openvla/openvla-7b")
    ft_root, ft_map = tensors(CHECKPOINTS[args.suite])
    sq_diff, sq_pub = defaultdict(float), defaultdict(float)
    only_ft = [k for k in ft_map if k not in pub_map]
    ranks = {}
    for key in sorted(k for k in ft_map if k in pub_map):
        with safe_open(pub_root / pub_map[key], framework="pt") as f:
            w_pub = f.get_tensor(key).float()
        with safe_open(ft_root / ft_map[key], framework="pt") as f:
            w_ft = f.get_tensor(key).float()
        if w_pub.shape != w_ft.shape:
            only_ft.append(f"{key} (shape {tuple(w_pub.shape)} -> {tuple(w_ft.shape)})")
            continue
        d = w_ft - w_pub
        part = part_of(key)
        sq_diff[part] += float(d.pow(2).sum())
        sq_pub[part] += float(w_pub.pow(2).sum())
        if (part.startswith("llm") and int(part[3:]) in args.rank_layers and key.endswith("proj.weight")) \
                or key in ("projector.fc1.weight", "vision_backbone.featurizer.blocks.0.attn.qkv.weight"):
            s = torch.linalg.svdvals(d)
            ranks[key] = {
                "rel_change": float(d.norm() / w_pub.norm()),
                "top_singular": [round(float(x), 4) for x in s[:40]],
                "energy_in_top32": float(s[:32].pow(2).sum() / s.pow(2).sum()),
            }
            print(f"{key}: rel {ranks[key]['rel_change']:.4f}, energy in top 32 SVs {ranks[key]['energy_in_top32']:.3f}",
                  flush=True)
    rel = {part: (sq_diff[part] / sq_pub[part]) ** 0.5 for part in sorted(sq_diff)}
    for part, r in rel.items():
        print(f"{part:45s} {r:.5f}")
    print("in fine-tuned only / changed shape:", only_ft)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"suite": args.suite, "rel_change": rel, "ranks": ranks,
                                          "only_in_finetuned": only_ft}, indent=2))


if __name__ == "__main__":
    main()
