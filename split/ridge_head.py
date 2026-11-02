"""Fit Model 2 as one linear map by closed-form ridge regression on observed (hidden state, action) pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

STEP_DIM = 7 * 4096


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--test_from", type=int, required=True)
    p.add_argument("--test_to", type=int, required=True)
    p.add_argument("--val_from", type=int, required=True)
    p.add_argument("--val_to", type=int, required=True)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    p.add_argument("--max_queries", type=int, default=None, help="Use only this many pool queries (data-efficiency); lambda is then chosen on ~20%% of those episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load(pairs_dir, keep):
    hs, acts, n_eps = [], [], 0
    for f in sorted(Path(pairs_dir).glob("task*_ep*.pt")):
        e = torch.load(f)
        if keep(e["episode"]):
            hs.append(e["hidden"].reshape(-1, 8, STEP_DIM))
            acts.append(e["actions"])
            n_eps += 1
    return hs, acts, n_eps


def to_steps(hs, acts):
    """Per-query lists -> (samples, 28672) float32 inputs and (samples, 7) float64 targets."""
    x = torch.cat(hs).reshape(-1, STEP_DIM).float()
    y = torch.cat(acts).reshape(-1, 7).double()
    return x, y


class Ridge:
    """Ridge on standardized features via one eigendecomposition of the Gram matrix."""

    def __init__(self, x, y, device, chunk=8192):
        n = x.shape[0]
        s = torch.zeros(STEP_DIM, dtype=torch.float64, device=device)
        ss = torch.zeros_like(s)
        for i in range(0, n, chunk):
            c = x[i:i + chunk].to(device).double()
            s += c.sum(0)
            ss += (c * c).sum(0)
        self.mu = (s / n).float()
        self.sd = (ss / n - (s / n) ** 2).clamp_min(1e-12).sqrt().float().clamp_min(1e-6)
        self.y_mean = y.mean(0)
        yc = (y - self.y_mean).to(device)
        gram = torch.zeros(STEP_DIM, STEP_DIM, dtype=torch.float32, device=device)
        xty = torch.zeros(STEP_DIM, 7, dtype=torch.float64, device=device)
        for i in range(0, n, chunk):
            c = (x[i:i + chunk].to(device) - self.mu) / self.sd
            gram += c.T @ c
            xty += c.T.double() @ yc[i:i + chunk]
        evals, evecs = torch.linalg.eigh(gram)
        del gram
        self.evals = evals.double().cpu()
        self.evecs = evecs.cpu()
        self.proj = self.evecs.double().T @ xty.cpu()  # (d, 7)
        self.mu, self.sd = self.mu.cpu(), self.sd.cpu()

    def weights(self, lam):
        """Weights and bias in the ORIGINAL input space: y = x @ w + b."""
        ws = self.evecs.double() @ (self.proj / (self.evals + lam)[:, None])
        w = ws / self.sd.double()[:, None]
        b = self.y_mean - self.mu.double() @ w
        return w, b


def metrics(x, y, w, b):
    pred = x.double() @ w + b
    l1 = (pred[:, :6] - y[:, :6]).abs().mean(0)
    grip = ((pred[:, 6] > 0) == (y[:, 6] > 0)).double().mean()
    return {"l1_per_dim": l1.tolist(), "l1_mean": float(l1.mean()), "gripper_accuracy": float(grip)}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    in_test = lambda ep: args.test_from <= ep < args.test_to
    in_val = lambda ep: args.val_from <= ep < args.val_to

    pool_h, pool_a, n_pool = load(args.pairs_dir, lambda ep: not in_test(ep))
    if args.max_queries:  # whole episodes, fixed shuffled order
        order = np.random.default_rng(args.seed).permutation(len(pool_h))
        kept_h, kept_a, n = [], [], 0
        for i in order:
            if n >= args.max_queries:
                break
            k = min(len(pool_h[i]), args.max_queries - n)
            kept_h.append(pool_h[i][:k]); kept_a.append(pool_a[i][:k]); n += k
        pool_h, pool_a = kept_h, kept_a
    test_h, test_a, n_test = load(args.pairs_dir, in_test)

    # Choose lambda on a validation split of the pool
    if args.max_queries:
        # last ~20% of the observed episodes
        n_val = max(1, round(0.2 * len(pool_h))) if len(pool_h) >= 2 else 0
        sel_h, sel_a = pool_h[:len(pool_h) - n_val], pool_a[:len(pool_a) - n_val]
        val_h, val_a = pool_h[len(pool_h) - n_val:], pool_a[len(pool_a) - n_val:]
    else:
        files = sorted(Path(args.pairs_dir).glob("task*_ep*.pt"))
        val_mask = [in_val(int(f.stem.split("_ep")[1])) for f in files if not in_test(int(f.stem.split("_ep")[1]))]
        sel_h = [h for h, v in zip(pool_h, val_mask) if not v]
        sel_a = [a for a, v in zip(pool_a, val_mask) if not v]
        val_h = [h for h, v in zip(pool_h, val_mask) if v]
        val_a = [a for a, v in zip(pool_a, val_mask) if v]

    lam_scores = {}
    if val_h:
        x_tr, y_tr = to_steps(sel_h, sel_a)
        x_va, y_va = to_steps(val_h, val_a)
        r = Ridge(x_tr, y_tr, args.device)
        for lam in args.lambdas:
            w, b = r.weights(lam)
            lam_scores[lam] = metrics(x_va, y_va, w, b)["l1_mean"]
        best = min(lam_scores, key=lam_scores.get)
        del x_tr, x_va, r
    else:
        best = args.lambdas[len(args.lambdas) // 2]

    # Refit on the whole pool, score on the test episodes
    x, y = to_steps(pool_h, pool_a)
    r = Ridge(x, y, args.device)
    w, b = r.weights(best)
    x_te, y_te = to_steps(test_h, test_a)
    result = {"train": metrics(x, y, w, b), "test": metrics(x_te, y_te, w, b),
              "predict_mean_l1": float((y_te[:, :6] - y[:, :6].mean(0)).abs().mean())}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    linear = torch.nn.Linear(STEP_DIM, 7)
    with torch.no_grad():
        linear.weight.copy_(w.T.float())
        linear.bias.copy_(b.float())
    torch.save(linear.state_dict(), out / "head.pt")
    config = {
        "kind": "replacement", "arch": "linear", "fit": "ridge_closed_form",
        "pairs_dir": args.pairs_dir, "test_episodes": [args.test_from, args.test_to],
        "val_episodes": [args.val_from, args.val_to], "lambda_val_l1": lam_scores, "lambda": best,
        "num_train_queries": int(x.shape[0] // 8), "num_test_queries": int(x_te.shape[0] // 8),
        "params": STEP_DIM * 7 + 7, "fit_seconds": time.time() - t0,
        # outputs are in action units; gripper = sign(output 7)
        "action_mean": [0.0] * 6, "action_std": [1.0] * 6,
        "metrics": result,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"ridge lambda={best:g} (val L1 {lam_scores}) | train {int(x.shape[0] // 8)} queries | "
          f"test L1 {result['test']['l1_mean']:.4f} (predict-mean {result['predict_mean_l1']:.4f}) "
          f"gripper {result['test']['gripper_accuracy']:.4f} | {config['fit_seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
