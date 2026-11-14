"""Fit π0.5's action expert as a small MLP (standardized features -> 512 -> ReLU -> 35) on recorded traffic.

Same data, features and folds as ridge_pi05.py; only the map differs. AdamW, L1 on standardized targets. The number of
training steps plays the part of ridge's lambda: chosen on the validation starts, then the MLP is retrained on
train + validation and scored on the test starts. Pooled features are cached per suite (--cache).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ridge_pi05 import ACTION_DIM, STEPS, features, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--record_dir", required=True)
    p.add_argument("--cache", required=True, help="Pooled-feature cache for this suite (built on first use)")
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--test_from", type=int, required=True)
    p.add_argument("--test_to", type=int, required=True)
    p.add_argument("--val_from", type=int, required=True)
    p.add_argument("--val_to", type=int, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[3000, 6000, 12000, 24000, 48000])
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    # standardize: per-feature training-set statistics, as ridge_pi05.py. layernorm (per input vector) lost 7.8 pp
    # closed loop (RESULTS.md §15)
    p.add_argument("--norm", choices=["standardize", "layernorm"], default="standardize")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout on the hidden layer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_cached(record_dir, pool, cache):
    """All recorded queries of a suite: x (Q, D) bf16 features, y (Q, 35) float64 actions, ep (Q,) start index."""
    cache = Path(cache)
    if cache.exists():
        c = torch.load(cache)
        assert c["record_dir"] == str(record_dir) and c["pool"] == pool, (c["record_dir"], c["pool"])
        return c["x"], c["y"], c["episode"]
    xs, ys, eps = [], [], []
    for f in sorted(Path(record_dir).glob("task*_ep*.pt")):
        e = torch.load(f, weights_only=False)
        xs.append(features(e["layer18_in"], pool))
        ys.append(e["actions"][:, :STEPS].reshape(len(e["actions"]), -1).double())
        eps.append(torch.full((len(e["actions"]),), e["episode"], dtype=torch.int64))
    x, y, ep = torch.cat(xs), torch.cat(ys), torch.cat(eps)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "episode": ep, "record_dir": str(record_dir), "pool": pool}, cache)
    return x, y, ep


class MLPMap(nn.Module):
    """Features -> 35 executed-action values (targets standardized inside, so it returns actions)."""

    def __init__(self, features_dim, hidden, y_mean, y_std, norm="layernorm", dropout=0.0):
        super().__init__()
        self.norm = norm
        layers = [nn.LayerNorm(features_dim)] if norm == "layernorm" else []
        layers += [nn.Linear(features_dim, hidden), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        self.net = nn.Sequential(*layers, nn.Linear(hidden, STEPS * ACTION_DIM))
        self.register_buffer("y_mean", y_mean.float())
        self.register_buffer("y_std", y_std.float())
        if norm == "standardize":
            self.register_buffer("x_mean", torch.zeros(features_dim))
            self.register_buffer("x_std", torch.ones(features_dim))

    @classmethod
    def from_config(cls, config):
        n = config["outputs"]
        return cls(config["features"], config["hidden"], torch.zeros(n), torch.ones(n),
                   config.get("norm", "layernorm"), config.get("dropout", 0.0))  # early fits saved no "norm"

    def standardized(self, x):
        x = x.float()
        if self.norm == "standardize":
            x = (x - self.x_mean) / self.x_std
        return self.net(x)

    def forward(self, x):
        return self.standardized(x) * self.y_std + self.y_mean


def train(x, y, steps, args, eval_x=None):
    torch.manual_seed(args.seed)
    dev = args.device
    y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-6)
    model = MLPMap(x.shape[1], args.hidden, y_mean, y_std, args.norm, args.dropout)
    if args.norm == "standardize":  # in row blocks: a full float copy would be ~6 GB
        s = torch.zeros(x.shape[1], dtype=torch.float64)
        ss = torch.zeros_like(s)
        for i in range(0, len(x), 2048):
            c = x[i:i + 2048].double()
            s += c.sum(0)
            ss += (c * c).sum(0)
        mean = s / len(x)
        model.x_mean.copy_(mean.float())
        model.x_std.copy_((ss / len(x) - mean**2).clamp_min(1e-12).sqrt().float().clamp_min(1e-6))
    model = model.to(dev)
    xg, target = x.to(dev), ((y - y_mean) / y_std).float().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(xg), (args.batch_size,), device=dev, generator=gen)
        loss = F.l1_loss(model.standardized(xg[idx]), target[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    del xg, target
    return model


@torch.no_grad()
def predict(model, x, rows=1024):
    dev = next(model.parameters()).device
    return torch.cat([model(x[i:i + rows].to(dev)).double().cpu() for i in range(0, len(x), rows)])


def main():
    args = parse_args()
    t0 = time.time()
    x_all, y_all, ep = load_cached(args.record_dir, args.pool, args.cache)
    test = (ep >= args.test_from) & (ep < args.test_to)
    val = (ep >= args.val_from) & (ep < args.val_to)
    sel = ~test & ~val
    assert not (test & val).any()
    print(f"features {x_all.shape[1]:,} | select on {int(sel.sum())} queries, validate on {int(val.sum())}, "
          f"test {int(test.sum())} | load {time.time() - t0:.0f}s", flush=True)

    step_scores = {}
    for s in args.steps:
        model = train(x_all[sel], y_all[sel], s, args)
        step_scores[s] = metrics(predict(model, x_all[val]), y_all[val])["l1_mean"]
        print(f"  steps {s}: validation L1 {step_scores[s]:.4f}", flush=True)
        del model
    best = min(step_scores, key=step_scores.get)

    fit = sel | val
    t1 = time.time()
    model = train(x_all[fit], y_all[fit], best, args)
    train_seconds = time.time() - t1
    result = {"train": metrics(predict(model, x_all[fit]), y_all[fit]),
              "test": metrics(predict(model, x_all[test]), y_all[test]),
              "predict_mean_l1": float((y_all[test].reshape(-1, STEPS, ACTION_DIM)[..., :6]
                                        - y_all[fit].reshape(-1, STEPS, ACTION_DIM)[..., :6].mean((0, 1))).abs().mean())}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "mlp.pt")
    config = {
        "kind": "pi05_mlp", "fit": "adamw_l1_steps_on_validation", "record_dir": args.record_dir,
        "layer": "layer18_in", "pool": args.pool, "features": int(x_all.shape[1]), "hidden": args.hidden,
        "outputs": STEPS * ACTION_DIM, "test_episodes": [args.test_from, args.test_to],
        "val_episodes": [args.val_from, args.val_to], "steps_val_l1": step_scores, "steps": best,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "norm": args.norm, "dropout": args.dropout,
        "num_train_queries": int(fit.sum()), "num_test_queries": int(test.sum()),
        "params": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds,
        "fit_seconds": time.time() - t0, "metrics": result,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"mlp pool {args.pool}: steps={best} | train {config['num_train_queries']} queries | "
          f"test L1 {result['test']['l1_mean']:.4f} (predict-mean {result['predict_mean_l1']:.4f}) "
          f"gripper {result['test']['gripper_accuracy']:.4f} | {config['params']:,} params | "
          f"{config['fit_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
