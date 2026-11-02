"""Model 2: interface tensor -> executable actions (head, un-normalization, gripper)."""

import json
from pathlib import Path

import numpy as np
import torch

from head import ACTION_DIM, NUM_ACTIONS_CHUNK, L1RegressionActionHead


class Model2:
    def __init__(self, head, action_stats, device):
        self.head = head
        self.device = device
        self.action_high = np.array(action_stats["q99"])
        self.action_low = np.array(action_stats["q01"])
        self.mask = np.array(action_stats["mask"]) if "mask" in action_stats else np.ones_like(self.action_low, dtype=bool)
        self.num_queries = 0

    @classmethod
    def load(cls, bundle_dir, device="cuda:0"):
        bundle_dir = Path(bundle_dir)
        config = json.loads((bundle_dir / "config.json").read_text())
        if config.get("kind") == "replacement":
            # extraction-test replacement
            from replacement import ReplacementModel2

            return ReplacementModel2.load(bundle_dir, device)
        assert config["normalization_type"] == "bounds_q99", config["normalization_type"]
        head = L1RegressionActionHead(
            input_dim=config["llm_dim"], hidden_dim=config["llm_dim"], action_dim=config["action_dim"]
        )
        # same order as get_action_head
        head = head.to(torch.bfloat16).to(device)
        head.eval()
        head.load_state_dict(torch.load(bundle_dir / "head.pt", map_location=device, weights_only=True))
        action_stats = json.loads((bundle_dir / "action_stats.json").read_text())
        return cls(head, action_stats, device)

    def normalized_actions(self, hidden):
        """(1, 56, 4096) bf16 -> (8, 7) float32 normalized actions."""
        with torch.inference_mode():
            out = self.head.predict_action(hidden.to(self.device))
            return out.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float().cpu().detach().numpy()

    def unnormalize(self, normalized):
        return np.where(
            self.mask,
            0.5 * (normalized + 1) * (self.action_high - self.action_low + 1e-8) + self.action_low,
            normalized,
        )

    @staticmethod
    def postprocess_gripper(actions):
        """[0,1] gripper -> binarized [-1,+1], then flip sign (LIBERO: -1 = open, +1 = close)."""
        actions = actions.copy()
        actions[..., -1] = np.sign(2 * (actions[..., -1] - 0.0) / (1.0 - 0.0) - 1)
        actions[..., -1] = actions[..., -1] * -1.0
        return actions

    def __call__(self, hidden):
        """(1, 56, 4096) bf16 -> (8, 7) float64 actions, ready to execute."""
        self.num_queries += 1
        return self.postprocess_gripper(self.unnormalize(self.normalized_actions(hidden)))
