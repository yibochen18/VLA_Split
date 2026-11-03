"""Unix-socket server running π0.5 (openpi's PyTorch policy, unmodified), for the LIBERO harness run_pi05_eval.py.

Runs in openpi's environment:  vla_openpi python split/pi05_server.py --socket /tmp/vla_pi05.sock
Images arrive rotated but not resized; they get openpi's own resize, as in openpi's examples/libero/main.py.
Each query's noise comes from (seed, task, episode, query), so a run gives the same actions whichever episode it
starts from. With --record_dir, the backbone states the action expert reads are saved per episode (see Recorder).
"""

import argparse
import dataclasses
import json
import os
import socket
import sys
import time
from pathlib import Path

import jax
import numpy as np
import torch
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_client import image_tools

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_protocol import ACTION_SHAPE, MAGIC_END, MAGIC_QUERY, MAGIC_STATS, actions_to_bytes, encode_response, read_request  # noqa: E402
from ridge_pi05 import STEPS, features  # noqa: E402

RESIZE = 224  # openpi's LIBERO client resize
ACTION_DIM = 7
IMAGE_TOKENS = 2 * 256  # third-person + wrist camera; the third image slot is masked out for π0.5


class Recorder:
    """Saves, per query, the valid prefix tokens (2 x 256 image tokens, then the instruction tokens) at:
    - "layer18_in": the input to the last backbone layer, the deepest backbone state the action expert reads
      (its KV cache at layer l comes from the states entering layer l);
    - "final": the last layer's output after the final norm, which the action expert never uses.
    """

    def __init__(self, model, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lm = model.paligemma_with_expert.paligemma.language_model
        self.current, self.episode = {}, []
        lm.layers[-1].register_forward_pre_hook(self._layer_in, with_kwargs=True)
        lm.norm.register_forward_hook(lambda m, args, out: self.current.__setitem__("final", out[0].detach()))
        embed_prefix = model.embed_prefix

        def embed_prefix_recorded(*args, **kwargs):
            embs, pad_masks, att_masks = embed_prefix(*args, **kwargs)
            self.current["pad_mask"] = pad_masks.detach()
            return embs, pad_masks, att_masks

        model.embed_prefix = embed_prefix_recorded

    def _layer_in(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        self.current["layer18_in"] = hidden.detach()

    def add(self, actions):
        mask = self.current["pad_mask"][0]
        assert int(mask[:IMAGE_TOKENS].sum()) == IMAGE_TOKENS, "unexpected image-token layout"
        keep = {k: self.current[k][0][mask].to(torch.bfloat16).cpu() for k in ("layer18_in", "final")}
        self.episode.append({**keep, "actions": torch.from_numpy(actions)})
        self.current = {}

    def end_episode(self, meta):
        stack = {k: torch.stack([q[k] for q in self.episode]) for k in ("layer18_in", "final", "actions")}
        torch.save({**stack, "image_tokens": IMAGE_TOKENS, **meta}, self.out_dir / f"{meta['episode_key']}.pt")
        self.episode = []


class _Stop(Exception):
    pass


class LinearPolicy:
    """π0.5 with its action expert replaced by a linear map (ridge_pi05.py): same input transforms and backbone,
    stopped at the input of the last backbone layer (what the expert reads); layer 18 and the expert never run.
    Returns the map's 5 actions; rows 6-10 of the 10 x 7 reply are NaN (never executed)."""

    def __init__(self, policy, head_dir):
        config = json.loads((Path(head_dir) / "config.json").read_text())
        assert config["kind"] == "pi05_linear" and config["layer"] == "layer18_in", config
        head = torch.load(Path(head_dir) / "linear.pt")
        self.policy, self.model, self.pool = policy, policy._model, config["pool"]
        self.device = policy._pytorch_device
        self.weight, self.bias = head["weight"].T.to(self.device), head["bias"].to(self.device)
        self.params = self.weight.numel() + self.bias.numel()
        self._model = self.model  # for Server.stats
        self.captured = None
        lm = self.model.paligemma_with_expert.paligemma.language_model
        lm.layers[-1].register_forward_pre_hook(self._stop, with_kwargs=True)

    def _stop(self, module, args, kwargs):
        self.captured = args[0] if args else kwargs["hidden_states"]
        raise _Stop

    @torch.inference_mode()
    def infer(self, obs, noise=None):
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...], inputs)
        model = self.model
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(_model.Observation.from_dict(inputs), train=False)
        prefix_embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att_4d = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        try:
            model.paligemma_with_expert.forward(attention_mask=att_4d, position_ids=torch.cumsum(pad_masks, dim=1) - 1,
                                                past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        except _Stop:
            pass
        states = self.captured[0][pad_masks[0]].to(torch.bfloat16)[None]
        y = features(states, self.pool).float() @ self.weight + self.bias
        actions = np.full(ACTION_SHAPE, np.nan)
        actions[:STEPS] = y.reshape(STEPS, -1).double().cpu().numpy()
        return {"actions": actions}


class Server:
    def __init__(self, policy, seed, recorder, noise_samples=0, samples_dir=None):
        self.policy, self.seed, self.recorder = policy, seed, recorder
        # Extra action samples per query with other noise, never executed: how much π0.5's own actions vary
        self.noise_samples, self.samples_dir, self.samples = noise_samples, samples_dir, []
        self.num_queries = self.bytes_in = self.bytes_out = 0
        self.times = {"preprocess": [], "compute": []}

    def noise(self, meta, sample=0):
        key = [self.seed, meta["task_id"], meta["episode"], meta["query"]] + ([sample] if sample else [])
        rng = np.random.default_rng(key)
        return rng.standard_normal((self.policy._model.config.action_horizon, self.policy._model.config.action_dim),
                                   dtype=np.float32)

    def stats(self):
        model = self.policy._model
        return {
            "num_queries": self.num_queries,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "mean_ms": {k: 1e3 * sum(v) / len(v) for k, v in self.times.items() if v},
            "params": sum(p.numel() for p in model.parameters()),
            "linear_map_params": getattr(self.policy, "params", None),
            "action_expert_params": sum(p.numel() for p in model.paligemma_with_expert.gemma_expert.parameters()),
            "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
            "pid": os.getpid(),
        }

    def infer(self, meta, image, wrist_image, state):
        t0 = time.perf_counter()
        element = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, RESIZE, RESIZE)),
            "observation/state": state,
            "prompt": meta["prompt"],
        }
        t1 = time.perf_counter()
        actions = np.asarray(self.policy.infer(element, noise=self.noise(meta))["actions"], dtype=np.float64)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.times["preprocess"].append(t1 - t0)
        self.times["compute"].append(t2 - t1)
        self.num_queries += 1
        if self.recorder:
            self.recorder.add(actions)
        if self.noise_samples:
            extra = [self.policy.infer(element, noise=self.noise(meta, k))["actions"] for k in range(1, self.noise_samples + 1)]
            self.samples.append(np.stack([actions, *np.asarray(extra, dtype=np.float64)]))
        return actions

    def handle(self, conn):
        while True:
            try:
                magic, request_id, meta, image, wrist_image, state, n_in = read_request(conn)
            except ConnectionError:
                return
            t0 = time.perf_counter()
            self.bytes_in += n_in
            if magic == MAGIC_STATS:
                payload = json.dumps(self.stats()).encode()
            elif magic == MAGIC_END:
                if self.recorder:
                    self.recorder.end_episode(meta)
                if self.noise_samples:
                    Path(self.samples_dir).mkdir(parents=True, exist_ok=True)
                    np.save(Path(self.samples_dir) / f"{meta['episode_key']}.npy", np.stack(self.samples))
                    self.samples = []
                payload = b""
            else:
                assert magic == MAGIC_QUERY, magic
                payload = actions_to_bytes(self.infer(meta, image, wrist_image, state))
            reply = encode_response(request_id, payload, time.perf_counter() - t0)
            conn.sendall(reply)
            self.bytes_out += len(reply)


def load_policy(checkpoint, config_name, device):
    config = _config.get_config(config_name)
    # Uncompiled: max-autotune picks kernels by timing them, which can differ between runs on a shared GPU
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    return policy_config.create_trained_policy(config, checkpoint, pytorch_device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(os.environ.get("VLA_OPENPI", ""), "checkpoints/pi05_libero_pytorch"))
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--socket", default="/tmp/vla_pi05.sock")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=7, help="Base seed of the per-query noise")
    p.add_argument("--record_dir", default=None, help="Save the backbone states of every query, one file per episode")
    p.add_argument("--linear", default=None, help="ridge_pi05.py output: serve the linear map instead of the action expert")
    p.add_argument("--noise_samples", type=int, default=0, help="Also draw this many unexecuted samples per query")
    p.add_argument("--samples_dir", default=None, help="With --noise_samples: where to save them, one file per episode")
    args = p.parse_args()

    policy = load_policy(args.checkpoint, args.config, args.device)
    if args.linear:
        assert not (args.record_dir or args.noise_samples), "--linear runs neither the expert nor its sampling"
        policy = LinearPolicy(policy, args.linear)
    assert not (args.noise_samples and args.record_dir), "the extra samples would overwrite the recorded states"
    server = Server(policy, args.seed, Recorder(policy._model, args.record_dir) if args.record_dir else None,
                    args.noise_samples, args.samples_dir)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(args.socket)
    listener.listen(1)
    print(f"READY {args.socket}", flush=True)
    try:
        while True:
            conn, _ = listener.accept()
            with conn:
                server.handle(conn)
    finally:
        os.unlink(args.socket)


if __name__ == "__main__":
    main()
