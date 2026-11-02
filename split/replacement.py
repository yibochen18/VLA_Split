"""Replacement Model 2s (linear / mlp / oft) trained on observed traffic, with Model 2's call interface."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from head import ACTION_DIM, LLM_DIM, NUM_ACTIONS_CHUNK, MLPResNet

STEP_INPUT_DIM = ACTION_DIM * LLM_DIM  # 28,672


def build_network(arch):
    if arch == "linear":
        return nn.Linear(STEP_INPUT_DIM, ACTION_DIM)
    if arch == "mlp":
        return nn.Sequential(nn.LayerNorm(STEP_INPUT_DIM), nn.Linear(STEP_INPUT_DIM, 512), nn.ReLU(), nn.Linear(512, ACTION_DIM))
    if arch == "oft":
        # Same architecture as OFT's L1-regression head, randomly initialized
        return MLPResNet(num_blocks=2, input_dim=STEP_INPUT_DIM, hidden_dim=LLM_DIM, output_dim=ACTION_DIM)
    raise ValueError(arch)


def forward_steps(net, hidden):
    """(B, 56, 4096) -> (B, 8, 7) raw outputs."""
    return net(hidden.reshape(hidden.shape[0], NUM_ACTIONS_CHUNK, STEP_INPUT_DIM).float())


class ReplacementModel2:
    """Same call interface as model2.Model2: (1, 56, 4096) bf16 -> (8, 7) float64 executable actions."""

    def __init__(self, net, mean, std, device):
        self.head = net
        self.device = device
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.num_queries = 0

    @classmethod
    def load(cls, bundle_dir, device="cuda:0"):
        bundle_dir = Path(bundle_dir)
        config = json.loads((bundle_dir / "config.json").read_text())
        net = build_network(config["arch"]).to(device)
        net.load_state_dict(torch.load(bundle_dir / "head.pt", map_location=device, weights_only=True))
        net.eval()
        return cls(net, config["action_mean"], config["action_std"], device)

    def __call__(self, hidden):
        self.num_queries += 1
        with torch.inference_mode():
            out = forward_steps(self.head, hidden.to(self.device))[0].double().cpu().numpy()
        actions = np.empty_like(out)
        actions[:, :6] = out[:, :6] * self.std + self.mean
        actions[:, 6] = np.where(out[:, 6] > 0, 1.0, -1.0)
        return actions
