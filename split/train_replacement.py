"""Train a replacement Model 2 on N recorded (interface, action) pairs; starts >= --heldout_from are held out."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import paths
from replacement import build_network, forward_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", default=str(paths.EXTRACTION / "libero_spatial" / "pairs"))
    p.add_argument("--arch", choices=["linear", "mlp", "oft"], required=True)
    p.add_argument("--num_queries", type=int, required=True)
    p.add_argument("--heldout_from", type=int, default=40)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_root", default=str(paths.EXTRACTION / "libero_spatial" / "replacements"))
    return p.parse_args()


def load_episodes(pairs_dir):
    return [torch.load(f) for f in sorted(Path(pairs_dir).glob("task*_ep*.pt"))]


def stack(episodes):
    return torch.cat([e["hidden"] for e in episodes]), torch.cat([e["actions"] for e in episodes])


@torch.no_grad()
def evaluate(net, hidden, actions, mean, std, device, batch=256):
    preds = []
    for i in range(0, len(hidden), batch):
        preds.append(forward_steps(net, hidden[i:i + batch].to(device)).double().cpu())
    out = torch.cat(preds)
    pred_cont = out[..., :6] * std + mean
    l1 = (pred_cont - actions[..., :6]).abs().mean(dim=(0, 1))
    gripper_acc = ((out[..., 6] > 0) == (actions[..., 6] > 0)).double().mean()
    baseline_l1 = (mean - actions[..., :6]).abs().mean(dim=(0, 1))  # predict-the-mean baseline
    return {
        "l1_per_dim": l1.tolist(),
        "l1_mean": float(l1.mean()),
        "predict_mean_l1": float(baseline_l1.mean()),
        "gripper_accuracy": float(gripper_acc),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda:0"

    episodes = load_episodes(args.pairs_dir)
    pool = [e for e in episodes if e["episode"] < args.heldout_from]
    test = [e for e in episodes if e["episode"] >= args.heldout_from]
    random.Random(args.seed).shuffle(pool)
    train, n = [], 0
    for e in pool:
        if n >= args.num_queries:
            break
        k = min(len(e["hidden"]), args.num_queries - n)
        train.append({**e, "hidden": e["hidden"][:k], "actions": e["actions"][:k]})
        n += k
    assert n == args.num_queries, f"only {n} training queries available"

    train_h, train_a = stack(train)
    test_h, test_a = stack(test)
    mean = train_a[..., :6].reshape(-1, 6).mean(0)
    std = train_a[..., :6].reshape(-1, 6).std(0).clamp_min(1e-6)
    target_cont = ((train_a[..., :6] - mean) / std).float()
    target_grip = (train_a[..., 6] > 0).float()
    train_h = train_h.to(device)
    target_cont, target_grip = target_cont.to(device), target_grip.to(device)

    net = build_network(args.arch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    net.train()
    for step in range(args.steps):
        idx = torch.randint(0, len(train_h), (min(args.batch_size, len(train_h)),), device=device)
        out = forward_steps(net, train_h[idx])
        loss = F.l1_loss(out[..., :6], target_cont[idx]) + F.binary_cross_entropy_with_logits(out[..., 6], target_grip[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    train_seconds = time.time() - t0

    metrics = {
        "train": evaluate(net, train_h.cpu(), train_a, mean.double(), std.double(), device),
        "test": evaluate(net, test_h, test_a, mean.double(), std.double(), device),
    }

    out_dir = Path(args.out_root) / f"{args.arch}_n{args.num_queries}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / "head.pt")
    config = {
        "kind": "replacement",
        "arch": args.arch,
        "num_queries": args.num_queries,
        "num_train_episodes": len(train),
        "num_test_queries": len(test_h),
        "params": sum(p.numel() for p in net.parameters()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_seconds": train_seconds,
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "metrics": metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{args.arch} n={args.num_queries}: params {config['params']:,}, {train_seconds:.0f}s, "
          f"test L1 {metrics['test']['l1_mean']:.4f} (predict-mean {metrics['test']['predict_mean_l1']:.4f}), "
          f"gripper acc {metrics['test']['gripper_accuracy']:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
