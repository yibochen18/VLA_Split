"""A secret part of OFT and its recorded traffic; LoRA on public weights (attack 2, RESULTS §17).

Secret parts (names as in public_swap.py): "vision" + "projector" (input: pixels; output: projected patch
embeddings) and "llm:A-B" (input: hidden state before Llama layer A; output: after layer B), or vision + projector +
llm:1-B together. OFT's LLM input is BOS, 512 patch embeddings (2 images x 256), proprio token, prompt, 56 action
slots, stop token.
"""

import math
from pathlib import Path

import torch
from torch import nn

NUM_PATCH_TOKENS = 512


# --- Recorded traffic (run_split_eval.py --record_full): one file per query

def record_hooks(vla, layers, store):
    """Hooks that put each query's pixels, LLM input and all-token hidden states after `layers` into `store`."""
    def keep_pixels(module, inputs):
        store["pixel_values"] = inputs[0].detach().to(torch.bfloat16).cpu()

    def keep_llm(module, inputs, kwargs, output):
        store["inputs_embeds"] = kwargs["inputs_embeds"].detach().cpu()
        store["attention_mask"] = kwargs["attention_mask"].detach().cpu()
        store["h"] = {layer: output.hidden_states[layer].detach().cpu() for layer in layers}

    vla.vision_backbone.register_forward_pre_hook(keep_pixels)
    vla.language_model.register_forward_hook(keep_llm, with_kwargs=True)


def query_files(record_dir):
    return sorted(Path(record_dir).glob("task*_ep*_q*.pt"))


def parse_parts(parts):
    """-> (uses_vision, first_layer, last_layer); layers 1-based, None if no Llama layers."""
    vision = "vision" in parts
    assert vision == ("projector" in parts), "vision and projector are secret together"
    llm = [p for p in parts if p.startswith("llm:")]
    assert len(llm) <= 1
    if not llm:
        return vision, None, None
    a, b = (int(x) for x in llm[0][4:].split("-"))
    assert not vision or a == 1, "vision + layers must be contiguous"
    return vision, a, b


def load_query(path, parts, device):
    """Recorded query -> (what the secret part receives, what it returned)."""
    d = torch.load(path, weights_only=False)
    vision, a, b = parse_parts(parts)
    q = {"attention_mask": d["attention_mask"].to(device), "inputs_embeds": d["inputs_embeds"].to(device)}
    if vision:
        q["pixel_values"] = d["pixel_values"].to(device)
    else:
        q["h_in"] = q["inputs_embeds"] if a == 1 else d["h"][a - 1].to(device)
    target = d["h"][b] if b is not None else d["inputs_embeds"][:, 1:1 + NUM_PATCH_TOKENS]
    return q, target.to(device)



def secret_forward(vla, parts, q):
    """Secret part's input -> output for one recorded query (batch 1). With vision secret, the patch embeddings
    are recomputed from pixels; only the other input tokens come from the recording."""
    vision, a, b = parse_parts(parts)
    if vision:
        proj = vla.projector(vla.vision_backbone(q["pixel_values"]))
        if a is None:
            return proj
        e = q["inputs_embeds"]
        h = torch.cat([e[:, :1], proj, e[:, 1 + NUM_PATCH_TOKENS:]], dim=1)
    else:
        h = q["h_in"]
    return run_layers(vla, h, q["attention_mask"], a, b)


def run_layers(vla, h, mask, a, b):
    """Llama layers a..b on hidden state h through LlamaModel itself (same mask and positions as the real
    forward pass), with its layer list cut to a..b and the final norm skipped."""
    llama = vla.language_model.model
    layers, norm, n = llama.layers, llama.norm, llama.config.num_hidden_layers
    try:
        llama.layers, llama.norm, llama.config.num_hidden_layers = nn.ModuleList(layers[a - 1:b]), nn.Identity(), b - a + 1
        return llama(inputs_embeds=h, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    finally:
        llama.layers, llama.norm, llama.config.num_hidden_layers = layers, norm, n



class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) B A x; A, B in fp32, B starts at zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (self.scale * (x.float() @ self.lora_A.t() @ self.lora_B.t())).to(x.dtype)


def part_modules(vla, parts):
    """(name prefix, module) for each secret part."""
    out = []
    for part in parts:
        if part == "vision":
            out.append(("vision_backbone", vla.vision_backbone))
        elif part == "projector":
            out.append(("projector", vla.projector))
        elif part.startswith("llm:"):
            a, b = (int(x) for x in part[4:].split("-"))
            out += [(f"language_model.model.layers.{i - 1}", vla.language_model.model.layers[i - 1]) for i in range(a, b + 1)]
        else:
            raise ValueError(part)
    return out


def add_lora(vla, parts, rank=32, alpha=16):
    """Wrap every nn.Linear in the secret parts. Returns {module name: LoRALinear}."""
    wrapped = {}
    for prefix, root in part_modules(vla, parts):
        for name, child in list(root.named_modules()):
            for cname, sub in list(child.named_children()):
                if isinstance(sub, nn.Linear):
                    lora = LoRALinear(sub, rank, alpha)
                    setattr(child, cname, lora)
                    wrapped[".".join(x for x in (prefix, name, cname) if x)] = lora
    assert wrapped, parts
    return wrapped


def lora_state(wrapped):
    return {name: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu(), "scale": m.scale} for name, m in wrapped.items()}


def merge_lora(vla, state):
    """Add each saved scale * B A into its base weight, in place. Returns the number of layers merged."""
    modules = dict(vla.named_modules())
    with torch.no_grad():
        for name, f in state.items():
            lin = modules[name]
            assert isinstance(lin, nn.Linear), name
            delta = f["scale"] * (f["B"].float() @ f["A"].float())
            lin.weight.copy_((lin.weight.float() + delta.to(lin.weight.device)).to(lin.weight.dtype))
    return len(state)
