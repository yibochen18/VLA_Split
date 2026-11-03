"""Fit π0.5's action expert as one linear map by closed-form ridge regression on recorded (backbone state, action) pairs.

Same fit as ridge_head.py (standardized features, lambda chosen on a validation fold, refit, score on test starts), but
in dual form: with ~1M features and ~10k queries, the queries x queries Gram matrix is the small one.
Features per query (pi05_server.py's recordings): the 512 image tokens + the mean of the instruction tokens at one
recorded layer, optionally 4 x 4 average-pooled over each camera's 16 x 16 patch grid. Targets: the 5 executed actions.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

IMAGE_TOKENS, GRID, WIDTH = 512, 16, 2048
STEPS, ACTION_DIM = 5, 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--record_dir", required=True)
    p.add_argument("--layer", choices=["layer18_in", "final"], default="layer18_in")
    p.add_argument("--pool", type=int, default=1, help="Average-pool image tokens over pool x pool patches")
    p.add_argument("--test_from", type=int, required=True)
    p.add_argument("--test_to", type=int, required=True)
    p.add_argument("--val_from", type=int, required=True)
    p.add_argument("--val_to", type=int, required=True)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def features(states, pool):
    """(Q, tokens, 2048) recorded states -> (Q, D) bf16: image tokens (pooled) + mean instruction token."""
    q = states.shape[0]
    img = states[:, :IMAGE_TOKENS]
    if pool > 1:
        g = GRID // pool
        img = img.float().reshape(q, 2, g, pool, g, pool, WIDTH).mean((3, 5)).reshape(q, -1, WIDTH)
    text = states[:, IMAGE_TOKENS:].float().mean(1, keepdim=True)
    return torch.cat([img.float(), text], 1).reshape(q, -1).to(torch.bfloat16)


def load(record_dir, layer, pool, keep):
    xs, ys = [], []
    for f in sorted(Path(record_dir).glob("task*_ep*.pt")):
        e = torch.load(f, weights_only=False)
        if keep(e["episode"]):
            xs.append(features(e[layer], pool))
            ys.append(e["actions"][:, :STEPS].reshape(len(e["actions"]), -1))
    return torch.cat(xs), torch.cat(ys).double()


class DualRidge:
    """Ridge on standardized features, solved through the eigendecomposition of the n x n Gram matrix."""

    def __init__(self, x, y, device, chunk=32768, others=()):
        self.device, self.chunk = device, chunk
        n, d = x.shape
        s = torch.zeros(d, dtype=torch.float64, device=device)
        ss = torch.zeros_like(s)
        rows = max(1, 2**28 // d)  # ~2 GB of float64 per block
        for i in range(0, n, rows):
            c = x[i:i + rows].to(device).double()
            s += c.sum(0)
            ss += (c * c).sum(0)
        self.mu = (s / n).float()
        self.sd = (ss / n - (s / n) ** 2).clamp_min(1e-12).sqrt().float().clamp_min(1e-6)
        self.y_mean = y.mean(0)
        self.yc = (y - self.y_mean).to(device)
        # Gram matrix of the training queries, and cross-Gram matrices of any other sets (validation / test)
        self.gram = torch.zeros(n, n, dtype=torch.float64, device=device)
        self.cross = [torch.zeros(len(o), n, dtype=torch.float64, device=device) for o in others]
        for j in range(0, d, chunk):
            c = self._std(x, j)
            self.gram += (c @ c.T).double()
            for k, o in enumerate(others):
                self.cross[k] += (self._std(o, j) @ c.T).double()
        self.evals, self.evecs = torch.linalg.eigh(self.gram)
        self.proj = self.evecs.T @ self.yc

    def _std(self, x, j):
        sl = slice(j, j + self.chunk)
        return (x[:, sl].to(self.device).float() - self.mu[sl]) / self.sd[sl]

    def alpha(self, lam):
        return self.evecs @ (self.proj / (self.evals + lam)[:, None])

    def predict_other(self, k, lam):
        return (self.cross[k] @ self.alpha(lam)).cpu() + self.y_mean

    def weights(self, x, lam):
        """Weights and bias in the ORIGINAL feature space: y = x @ w + b."""
        a = self.alpha(lam).float()
        w = torch.empty(x.shape[1], a.shape[1])
        for j in range(0, x.shape[1], self.chunk):
            sl = slice(j, j + self.chunk)
            w[sl] = ((self._std(x, j).T @ a) / self.sd[sl, None]).cpu()
        b = self.y_mean - (self.mu.cpu().double() @ w.double())
        return w, b


def predict(x, w, b, rows=1024):
    return torch.cat([x[i:i + rows].float() @ w + b.float() for i in range(0, len(x), rows)]).double()


def metrics(pred, y):
    pred, y = pred.reshape(-1, STEPS, ACTION_DIM), y.reshape(-1, STEPS, ACTION_DIM)
    l1 = (pred[..., :6] - y[..., :6]).abs().mean((0, 1))
    grip = ((pred[..., 6] > 0) == (y[..., 6] > 0)).double().mean()
    return {"l1_per_dim": l1.tolist(), "l1_mean": float(l1.mean()), "gripper_accuracy": float(grip)}


def main():
    args = parse_args()
    t0 = time.time()
    in_test = lambda ep: args.test_from <= ep < args.test_to
    in_val = lambda ep: args.val_from <= ep < args.val_to
    x_sel, y_sel = load(args.record_dir, args.layer, args.pool, lambda ep: not in_test(ep) and not in_val(ep))
    x_val, y_val = load(args.record_dir, args.layer, args.pool, in_val)
    print(f"features {x_sel.shape[1]:,} | select on {len(x_sel)} queries, validate on {len(x_val)} | load {time.time() - t0:.0f}s", flush=True)

    r = DualRidge(x_sel, y_sel, args.device, others=[x_val])
    lam_scores = {lam: metrics(r.predict_other(0, lam), y_val)["l1_mean"] for lam in args.lambdas}
    best = min(lam_scores, key=lam_scores.get)
    print(f"validation L1 per lambda: {lam_scores} -> {best:g}", flush=True)
    del r

    x = torch.cat([x_sel, x_val])
    y = torch.cat([y_sel, y_val])
    del x_sel, x_val
    x_te, y_te = load(args.record_dir, args.layer, args.pool, in_test)
    r = DualRidge(x, y, args.device, others=[x_te])
    w, b = r.weights(x, best)
    result = {"train": metrics(predict(x, w, b), y), "test": metrics(predict(x_te, w, b), y_te),
              "test_via_gram": metrics(r.predict_other(0, best), y_te),
              "predict_mean_l1": float((y_te.reshape(-1, STEPS, ACTION_DIM)[..., :6]
                                        - y.reshape(-1, STEPS, ACTION_DIM)[..., :6].mean((0, 1))).abs().mean())}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": w.T.contiguous(), "bias": b.float()}, out / "linear.pt")
    config = {
        "kind": "pi05_linear", "fit": "ridge_closed_form_dual", "record_dir": args.record_dir,
        "layer": args.layer, "pool": args.pool, "features": int(x.shape[1]), "outputs": STEPS * ACTION_DIM,
        "test_episodes": [args.test_from, args.test_to], "val_episodes": [args.val_from, args.val_to],
        "lambda_val_l1": lam_scores, "lambda": best, "num_train_queries": len(x), "num_test_queries": len(x_te),
        "params": int(w.numel() + b.numel()), "fit_seconds": time.time() - t0, "metrics": result,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{args.layer} pool {args.pool}: lambda={best:g} | train {len(x)} queries | test L1 {result['test']['l1_mean']:.4f} "
          f"(predict-mean {result['predict_mean_l1']:.4f}) gripper {result['test']['gripper_accuracy']:.4f} | "
          f"{config['fit_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
