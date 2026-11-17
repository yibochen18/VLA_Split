"""Attack 1 (public-weights substitution): replace parts of the fine-tuned OFT model with the public
openvla/openvla-7b weights it was LoRA-fine-tuned from, as an attacker who lacks those parts would.

Parts: "vision" (both vision encoders), "projector", "llm:A-B" (Llama layers A..B, 1-based, inclusive).
The proprio projector and the action head have no public counterpart and are never swapped."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

PUBLIC_REPO = "openvla/openvla-7b"


def _prefixes(part):
    if part == "vision":
        return ["vision_backbone."]
    if part == "projector":
        return ["projector."]
    if part.startswith("llm:"):
        a, b = (int(x) for x in part[4:].split("-"))
        assert 1 <= a <= b <= 32, part
        return [f"language_model.model.layers.{i - 1}." for i in range(a, b + 1)]
    raise ValueError(f"unknown part {part!r}")


def swap_public(vla, parts):
    """Copy the public weights for `parts` into `vla` in place. Returns {part: number of tensors swapped}."""
    root = Path(snapshot_download(PUBLIC_REPO, allow_patterns=["*.json", "*.safetensors"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    own = vla.state_dict()
    counts = {}
    for part in parts:
        prefixes = _prefixes(part)
        keys = [k for k in weight_map if any(k.startswith(p) for p in prefixes)]
        assert keys, part
        for k in keys:
            assert k in own, f"public key {k} not in fine-tuned model"
            with safe_open(root / weight_map[k], framework="pt") as f:
                t = f.get_tensor(k)
            assert t.shape == own[k].shape, (k, t.shape, own[k].shape)
            with torch.no_grad():
                own[k].copy_(t.to(own[k].dtype))
        # every fine-tuned tensor of this part must have been replaced
        own_keys = [k for k in own if any(k.startswith(p) for p in prefixes)]
        assert sorted(own_keys) == sorted(keys), part
        counts[part] = len(keys)
    return counts
