"""Black box (RESULTS §17): no related weights; a replacement for secret Llama layers a..b is trained from scratch on
recorded (block input, block output) traffic, all tokens (fit_blackbox.py). Every replacement predicts the block's
output as h_in + update and starts as the identity:

  identity  the block dropped (control)
  linear    per-token linear map on standardized input (ridge)
  mlp       per-token MLP 4096 -> 8192 -> 4096
  tf<k>     k Llama-style layers at OFT's width, random init
  vit       for the vision encoder + projector (block "vision"): a small ViT from scratch, camera images -> the 512
            image tokens the LLM receives (no residual: input and output differ)

Served by hooks: the real block still runs and its output is replaced by the replacement's prediction, which is
the same as the replacement standing in for the block.
"""

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

D = 4096



class Identity(nn.Module):
    def forward(self, h):
        return h


class PerToken(nn.Module):
    """h_out = h + scale_out * g((h - mu) / sd); g linear or MLP."""

    def __init__(self, kind, hidden=8192):
        super().__init__()
        for name in ("mu", "sd", "scale_out"):
            self.register_buffer(name, torch.zeros(D) if name == "mu" else torch.ones(D))
        if kind == "linear":
            self.g = nn.Linear(D, D)
            nn.init.zeros_(self.g.weight)
            nn.init.zeros_(self.g.bias)
        else:
            self.g = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, D))
            nn.init.zeros_(self.g[2].weight)
            nn.init.zeros_(self.g[2].bias)

    def forward(self, h):
        z = (h.float() - self.mu) / self.sd
        return (h.float() + self.scale_out * self.g(z)).to(h.dtype)


class SmallTransformer(nn.Module):
    """k Llama decoder layers at OFT's width (4096, 32 heads, MLP 11008), random init; run through LlamaModel so
    the attention mask (bidirectional in OFT's fork) and rotary positions are handled as in the real model. Final norm skipped (residual stream)."""

    def __init__(self, n_layers):
        super().__init__()
        cfg = LlamaConfig(hidden_size=D, intermediate_size=11008, num_attention_heads=32, num_key_value_heads=32,
                          num_hidden_layers=n_layers, vocab_size=8, max_position_embeddings=2048,
                          rms_norm_eps=1e-6, attn_implementation="sdpa")
        self.model = LlamaModel(cfg)
        self.model.norm = nn.Identity()
        self.model.embed_tokens = None
        for layer in self.model.layers:
            nn.init.zeros_(layer.self_attn.o_proj.weight)
            nn.init.zeros_(layer.mlp.down_proj.weight)

    def forward(self, h):
        mask = torch.ones(h.shape[:2], dtype=torch.long, device=h.device)
        return self.model(inputs_embeds=h, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state


class VisionViT(nn.Module):
    """Per camera image (6 channels: OFT's two normalizations), 14x14 patches -> 256 tokens -> 4096; both images share
    the weights, as OFT's encoder does. Width 512, 8 layers, 8 heads (~27M parameters)."""

    def __init__(self, width=512, depth=8, heads=8, channels=6, patch=14, n_patches=256):
        super().__init__()
        self.channels = channels
        self.embed = nn.Conv2d(channels, width, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, n_patches, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, D)

    def forward(self, px):
        toks = []
        for img in torch.split(px, self.channels, dim=1):
            x = self.embed(img).flatten(2).transpose(1, 2) + self.pos
            toks.append(self.out(self.norm(self.blocks(x))))
        return torch.cat(toks, dim=1)



def build(kind):
    if kind == "identity":
        return Identity()
    if kind in ("linear", "mlp"):
        return PerToken(kind)
    if kind.startswith("tf"):
        return SmallTransformer(int(kind[2:]))
    if kind == "vit":
        return VisionViT()
    raise ValueError(kind)


def save(path, kind, block, module, extra):
    torch.save({"kind": kind, "block": block, "state_dict": module.state_dict(), **extra}, path)


def load(path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = build(d["kind"])
    m.load_state_dict(d["state_dict"])
    m.to(device).eval()
    if d["kind"].startswith("tf"):
        m.to(torch.bfloat16)
    return m, (d["block"] if d["block"] == "vision" else tuple(d["block"])), d


def install_hooks(vla, module, block):
    """Replace the output of Llama layers block[0]..block[1] (1-based) with module(input to block[0]); for block
    "vision", the projector's output (image tokens) with module(pixels)."""
    if block == "vision":
        return install_vision_hooks(vla, module)
    a, b = block
    layers = vla.language_model.model.layers
    held = {}

    def keep_input(mod, args, kwargs):
        held["h"] = args[0] if args else kwargs["hidden_states"]

    @torch.no_grad()
    def replace_output(mod, args, kwargs, output):
        h_in = held.pop("h")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_out = module(h_in)
        return (h_out.to(output[0].dtype),) + tuple(output[1:])

    return [layers[a - 1].register_forward_pre_hook(keep_input, with_kwargs=True),
            layers[b - 1].register_forward_hook(replace_output, with_kwargs=True)]


def install_vision_hooks(vla, module):
    held = {}

    def keep_pixels(mod, args):
        held["px"] = args[0]

    @torch.no_grad()
    def replace_output(mod, args, output):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(held.pop("px"))
        return out.to(output.dtype)

    return [vla.vision_backbone.register_forward_pre_hook(keep_pixels),
            vla.projector.register_forward_hook(replace_output)]



def load_pairs(files, block):
    """(h_in, h_out) per recorded query, bf16 on CPU. Block input: layer a-1's output (0 = LLM input). Block
    "vision": (pixels, the 512 image tokens of the LLM input)."""
    pairs = []
    if block == "vision":
        for f in files:
            d = torch.load(f, weights_only=False)
            pairs.append((d["pixel_values"][0].contiguous(), d["inputs_embeds"][0, 1:513].contiguous()))
        return pairs
    a, b = block
    for f in files:
        d = torch.load(f, weights_only=False)
        h_in = d["inputs_embeds"] if a == 1 else d["h"][a - 1]
        pairs.append((h_in[0].contiguous(), d["h"][b][0].contiguous()))
    return pairs
