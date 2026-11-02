"""What fits in a CPU TEE: time per query of one Llama-2-7B layer, the MLP head and the linear map on CPU (fp32).

CPU only, no GPU needed. The Llama layer has random weights (its time does not depend on them) and runs on OFT's
real sequence length: the prompt tokens of the suite's instructions, plus vision patches, proprio, action slots and
the stop token. Run from openvla-oft/:  python ../split/bench_tee.py --out $VLA_DATA/linear_compute/bench/tee.json
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from libero.libero import benchmark
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from bench_depth import stats, timed  # noqa: E402
from head import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from model2 import Model2  # noqa: E402
from verify_offline import CHECKPOINTS  # noqa: E402

NUM_PATCHES = 2 * 256 + 1  # two 224px images at patch 14, plus the proprio token
STOP = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--reps", type=int, default=5, help="Calls per Llama-layer timing (heads: 20x more)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def sequence_lengths(suite):
    """Tokens the Llama layers see per query, for each task of the suite (as in OFT's predict_action)."""
    tok = AutoTokenizer.from_pretrained(CHECKPOINTS[suite], trust_remote_code=True)
    lengths = []
    for task in [benchmark.get_benchmark_dict()[suite]().get_task(i) for i in range(10)]:
        ids = tok(f"In: What action should the robot take to {task.language.lower()}?\nOut:").input_ids
        prompt = len(ids) + (ids[-1] != 29871)  # predict_action appends the empty token 29871 if missing
        lengths.append(prompt + NUM_PATCHES + ACTION_DIM * NUM_ACTIONS_CHUNK + STOP)
    return lengths


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths = sequence_lengths(args.suite)
    seq = max(lengths)
    print(f"sequence length: {min(lengths)}-{seq} tokens, timing at {seq}", flush=True)

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=4096, intermediate_size=11008, num_attention_heads=32, num_key_value_heads=32)
    layer = LlamaDecoderLayer(cfg, layer_idx=0).float().eval()
    hidden = torch.randn(1, seq, 4096)
    position_ids = torch.arange(seq).unsqueeze(0)
    mlp = Model2.load(paths.SPLIT / args.suite / "model2", "cpu")
    mlp.head.float()
    linear = Model2.load(paths.DATA / "linear_head" / "heads" / f"{args.suite}_fold4", "cpu")
    x = torch.randn(1, 56, 4096)
    timings = {
        "llama_layer": (lambda _: layer(hidden, position_ids=position_ids), args.reps),
        "mlp_head": (lambda _: mlp(x), 20 * args.reps),
        "linear_map": (lambda _: linear(x), 20 * args.reps),
    }
    params = {"llama_layer": sum(p.numel() for p in layer.parameters()),
              "mlp_head": sum(p.numel() for p in mlp.head.parameters()),
              "linear_map": sum(p.numel() for p in linear.head.parameters())}

    result = {"suite": args.suite, "sequence_lengths": lengths, "timed_sequence_length": seq, "dtype": "float32",
              "cpu": platform.processor() or platform.machine(), "params": params, "times": {}}
    with torch.inference_mode():
        for threads in args.threads:
            torch.set_num_threads(threads)
            for name, (fn, reps) in timings.items():
                fn(None)
                result["times"][f"{name}_{threads}t"] = s = stats(timed(fn, reps, cuda=False))
                print(f"{threads}t {name}: {s['median_ms']:.2f} ms", flush=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
