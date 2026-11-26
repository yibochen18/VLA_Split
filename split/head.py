"""Standalone copy of OpenVLA-OFT's L1-regression action head (torch only)."""

import torch
import torch.nn as nn

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
LLM_DIM = 4096


class MLPResNetBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """Maps the 56 action-slot hidden states to an 8x7 chunk of normalized actions."""

    def __init__(self, input_dim=LLM_DIM, hidden_dim=LLM_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim * ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # (B, 56, D) -> (B, 8, 7 * D) -> (B, 8, 7)
        batch_size = actions_hidden_states.shape[0]
        return self.model(actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1))
