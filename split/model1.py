"""Model 1: images + instruction + state -> hidden states at the action positions.

Runs the stock predict_action with InterfaceCapture as the head. `layers` also returns earlier layers;
`depth` runs only the first `depth` Llama layers."""

import torch
from torch import nn

from experiments.robot.openvla_utils import get_processor, get_proprio_projector, get_vla_action
from experiments.robot.robot_utils import get_model


class _InterfaceReady(Exception):
    def __init__(self, hidden):
        super().__init__("interface tensor captured")
        self.hidden = hidden


class InterfaceCapture:
    """Stands in for the action head; predict_action calls `predict_action(actions_hidden_states)` on it."""

    def predict_action(self, actions_hidden_states):
        raise _InterfaceReady(actions_hidden_states)


def _action_slots(all_hidden, actions_hidden_states, layers):
    """Slice the action positions out of each requested layer's output, at the offset predict_action used."""
    last = all_hidden[-1]
    # start position from the view's storage offset
    start = (actions_hidden_states.storage_offset() - last.storage_offset()) // last.shape[-1]
    end = start + actions_hidden_states.shape[1]
    assert torch.equal(last[:, start:end], actions_hidden_states)
    return {layer: all_hidden[layer][:, start:end].clone() for layer in layers}


class Model1:
    def __init__(self, cfg, vla, processor, proprio_projector, layers=()):
        self.cfg = cfg
        self.layers = tuple(layers)
        self._all_hidden = None
        self.vla = vla
        self.processor = processor
        self.proprio_projector = proprio_projector
        self._capture = InterfaceCapture()
        if self.layers:
            vla.language_model.register_forward_hook(self._keep_hidden_states)
        llama = vla.language_model.model
        self._all_layers, self._norm, self._num_layers = llama.layers, llama.norm, len(llama.layers)
        self.depth = self._num_layers

    def set_depth(self, depth):
        """Run only Llama layers 1..depth; below the full depth the final norm is skipped (as in output_hidden_states)."""
        llama = self.vla.language_model.model
        llama.layers = nn.ModuleList(self._all_layers[:depth])
        llama.norm = self._norm if depth == self._num_layers else nn.Identity()
        llama.config.num_hidden_layers = depth
        self.depth = depth

    def _keep_hidden_states(self, module, inputs, output):
        self._all_hidden = output.hidden_states  # embeddings, then after each layer

    @classmethod
    def load(cls, cfg, layers=(), depth=None):
        """`cfg` is a run_libero_eval.GenerateConfig. Sets cfg.unnorm_key (used for the proprio stats)."""
        vla = get_model(cfg)
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
        processor = get_processor(cfg)
        # Same key resolution as check_unnorm_key in run_libero_eval.py
        suite = cfg.task_suite_name
        cfg.unnorm_key = suite if suite in vla.norm_stats else f"{suite}_no_noops"
        assert cfg.unnorm_key in vla.norm_stats, cfg.unnorm_key
        # action stats belong to Model 2
        for key in vla.norm_stats:
            vla.norm_stats[key].pop("action", None)
        model = cls(cfg, vla, processor, proprio_projector, layers)
        if depth is not None and depth < model._num_layers:
            assert not layers, "depth and layers are separate tests"
            model.set_depth(depth)
            model._all_layers = model._all_layers[:depth]  # free the removed layers
            torch.cuda.empty_cache()
        return model

    def __call__(self, obs, task_label):
        """Returns the cut tensor, or (cut tensor, {layer: tensor}) when loaded with `layers`."""
        try:
            get_vla_action(
                self.cfg, self.vla, self.processor, obs, task_label,
                action_head=self._capture, proprio_projector=self.proprio_projector,
            )
        except _InterfaceReady as ready:
            if not self.layers:
                return ready.hidden
            per_layer = _action_slots(self._all_hidden, ready.hidden, self.layers)
            self._all_hidden = None
            return ready.hidden, per_layer
        raise RuntimeError("predict_action returned without reaching the action head")
