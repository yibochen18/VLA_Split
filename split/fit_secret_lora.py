"""Attack 2 (RESULTS §17): rank-32 LoRA on the public openvla-7b weights of a secret part, fitted so it reproduces
the secret part's recorded output (all tokens) from its recorded input, on N observed queries.

Loss: squared error per feature, scaled by the feature's spread over the training targets (a few residual-stream
features are very large). Fixed recipe for every N. The `val` queries only report the error; nothing is chosen on them.

  cd openvla-oft && python ../split/fit_secret_lora.py --suite libero_spatial --parts llm:1-16 \
      --record_dir $VLA_DATA/secret_lora/record/libero_spatial --n_queries 1000 --out ...
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle
from experiments.robot.robot_utils import get_model
import paths
from public_swap import swap_public
from secret_lora import add_lora, load_query, lora_state, query_files, secret_forward
from verify_offline import CHECKPOINTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--record_dir", required=True)
    p.add_argument("--n_queries", type=int, required=True, help="Observed queries the attacker trains on")
    p.add_argument("--n_val", type=int, default=200, help="Recorded queries for the reported error only")
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=1500, help="Optimizer steps")
    p.add_argument("--batch", type=int, default=8, help="Queries per step (accumulated one at a time)")
    p.add_argument("--check_finetuned", action="store_true",
                   help="Check only: the fine-tuned weights must reproduce the recorded outputs")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def evaluate(vla, parts, files, scale, device):
    """Mean scaled MSE and relative error ||pred - y|| / ||y|| over `files`."""
    mse, rel = [], []
    for f in files:
        q, y = load_query(f, parts, device)
        pred = secret_forward(vla, parts, q).float()
        y = y.float()
        mse.append(float((((pred - y) / scale) ** 2).mean()))
        rel.append(float((pred - y).norm() / y.norm()))
    return sum(mse) / len(mse), sum(rel) / len(rel)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)

    order = query_files(args.record_dir)
    random.Random(args.subset_seed).shuffle(order)
    assert len(order) >= args.n_queries + args.n_val, (len(order), args.n_queries, args.n_val)
    train, val = order[:args.n_queries], order[-args.n_val:]

    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    vla = get_model(cfg)
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)

    # per-feature scale of the targets (training queries only)
    sq, s, n = 0.0, 0.0, 0
    for f in train[:min(len(train), 500)]:
        _, y = load_query(f, args.parts, device)
        y = y.float().reshape(-1, y.shape[-1])
        s, sq, n = s + y.sum(0), sq + (y ** 2).sum(0), n + y.shape[0]
    scale = ((sq / n - (s / n) ** 2).clamp_min(0) ** 0.5).clamp_min(1e-3)

    if args.check_finetuned:
        mse, rel = evaluate(vla, args.parts, val[:20], scale, device)
        print(f"fine-tuned weights vs recorded: scaled MSE {mse:.3e}, relative error {rel:.3e}")
        (out / "check_finetuned.json").write_text(json.dumps({"mse": mse, "rel": rel}, indent=2))
        return

    swapped = swap_public(vla, args.parts)
    mse0, rel0 = evaluate(vla, args.parts, val, scale, device)
    print(f"public weights: val scaled MSE {mse0:.4f}, relative error {rel0:.4f}", flush=True)

    wrapped = add_lora(vla, args.parts, args.rank, args.alpha)
    params = [p for m in wrapped.values() for p in (m.lora_A, m.lora_B)]
    print(f"LoRA on {len(wrapped)} linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / 50) * 0.5 * (1 + torch.cos(torch.tensor(step / args.steps * torch.pi)).item()))

    log = []
    t0 = time.time()
    sample = random.Random(1)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.batch):
            q, y = load_query(sample.choice(train), args.parts, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = secret_forward(vla, args.parts, q)
            loss = (((pred.float() - y.float()) / scale) ** 2).mean() / args.batch
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step} loss {total:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            log.append({"step": step, "loss": total})

    vla.eval()
    mse1, rel1 = evaluate(vla, args.parts, val, scale, device)
    print(f"fitted: val scaled MSE {mse1:.4f}, relative error {rel1:.4f}", flush=True)
    torch.save({"lora": lora_state(wrapped), "parts": args.parts}, out / "lora.pt")
    (out / "fit.json").write_text(json.dumps({
        "suite": args.suite, "parts": args.parts, "swapped": swapped, "n_queries": args.n_queries,
        "n_val": len(val), "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "steps": args.steps,
        "batch": args.batch, "lora_params": sum(p.numel() for p in params),
        "val_public": {"mse": mse0, "rel": rel0}, "val_fitted": {"mse": mse1, "rel": rel1},
        "train_minutes": (time.time() - t0) / 60, "log": log,
    }, indent=2))


if __name__ == "__main__":
    main()
