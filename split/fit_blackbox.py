"""Fit a from-scratch replacement for secret Llama layers a..b (blackbox.py) on N observed queries.

Budget: N queries in all. 20% of them (at least 20) are the attacker's own held-out set ("sel") for choosing lambda
(linear; then refit on all N) or the best step (mlp, tf<k>); the rest train. 200 other recorded queries only report
the error (squared error per feature, scaled by the target's spread).

  python split/fit_blackbox.py --block 9 16 --kind tf2 --record_dir $VLA_DATA/secret_lora/record/libero_spatial \
      --n_queries 1000 --out .../L9-16_tf2_n1000.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blackbox  # noqa: E402
from secret_lora import query_files  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, nargs=2, required=True, help="First and last secret Llama layer (1-based)")
    p.add_argument("--kind", required=True, help="identity, linear, mlp, tf2, tf4")
    p.add_argument("--record_dir", required=True)
    p.add_argument("--n_queries", type=int, required=True)
    p.add_argument("--n_report", type=int, default=200, help="Recorded queries for the reported error only")
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=4, help="Queries per step")
    p.add_argument("--lr", type=float, default=None, help="Default 1e-4 (mlp), 3e-4 (tf)")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--out", required=True)
    return p.parse_args()


def scaled_mse(pred, y, scale):
    return (((pred.float() - y.float()) / scale) ** 2).mean()


@torch.no_grad()
def evaluate(module, pairs, scale, dev):
    module.eval()
    tot = 0.0
    for h_in, h_out in pairs:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = module(h_in.to(dev)[None])
        tot += float(scaled_mse(pred[0], h_out.to(dev), scale))
    return tot / len(pairs)


def feature_stats(pairs, dev, which):
    s = sq = 0.0
    n = 0
    for p in pairs[:500]:
        x = p[which].to(dev).float()
        s, sq, n = s + x.sum(0), sq + (x ** 2).sum(0), n + x.shape[0]
    mu = s / n
    return mu, ((sq / n - mu ** 2).clamp_min(0) ** 0.5).clamp_min(1e-3)


LAMBDAS = (1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)


@torch.no_grad()
def fit_ridge(module, train, sel, scale, dev, lams=LAMBDAS):
    """Ridge for the per-token linear map: (h_out - h_in) / scale_out from standardized h_in. With `sel`, keeps the
    lambda with the lowest error on it; without, fits the single lambda given."""
    xtx = torch.zeros(blackbox.D + 1, blackbox.D + 1, device=dev, dtype=torch.float64)
    xty = torch.zeros(blackbox.D + 1, blackbox.D, device=dev, dtype=torch.float64)
    for h_in, h_out in train:
        x = h_in.to(dev).float()
        z = torch.cat([(x - module.mu) / module.sd, torch.ones(len(x), 1, device=dev)], 1).double()
        t = ((h_out.to(dev).float() - x) / module.scale_out).double()
        xtx += z.T @ z
        xty += z.T @ t
    n_tok = sum(len(h) for h, _ in train)
    best = None
    for lam in lams:
        reg = torch.eye(blackbox.D + 1, device=dev, dtype=torch.float64) * lam * n_tok / 1000
        reg[-1, -1] = 0
        w = torch.linalg.solve(xtx + reg, xty)
        module.g.weight.copy_(w[:-1].T.float())
        module.g.bias.copy_(w[-1].float())
        if sel is None:
            return {"lambda": lam}
        err = evaluate(module, sel, scale, dev)
        print(f"lambda {lam}: sel {err:.4f}", flush=True)
        if best is None or err < best[0]:
            best = (err, lam, module.g.weight.clone(), module.g.bias.clone())
    module.g.weight.copy_(best[2])
    module.g.bias.copy_(best[3])
    return {"lambda": best[1]}


def fit_sgd(module, train, sel, scale, dev, args):
    """AdamW, warmup + cosine; keeps the step with the lowest error on `sel`."""
    lr = args.lr or (1e-4 if args.kind == "mlp" else 3e-4)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + torch.cos(torch.tensor(s / args.steps * torch.pi)).item()))
    rng = random.Random(1)
    best = (evaluate(module, sel, scale, dev), 0, {k: v.detach().clone() for k, v in module.state_dict().items()})
    log = [{"step": 0, "sel": best[0]}]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        module.train()
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.batch):
            h_in, h_out = rng.choice(train)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = module(h_in.to(dev, non_blocking=True)[None])
            loss = scaled_mse(pred[0], h_out.to(dev, non_blocking=True), scale) / args.batch
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            err = evaluate(module, sel, scale, dev)
            log.append({"step": step, "train": tot, "sel": err})
            print(f"step {step} train {tot:.4f} sel {err:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if err < best[0]:
                best = (err, step, {k: v.detach().clone() for k, v in module.state_dict().items()})
    module.load_state_dict(best[2])
    return {"lr": lr, "best_step": best[1], "log": log}


def main():
    args = parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    block = tuple(args.block)
    order = query_files(args.record_dir)
    random.Random(args.subset_seed).shuffle(order)  # same subsets as fit_secret_lora.py
    assert len(order) >= args.n_queries + args.n_report
    t0 = time.time()
    budget = blackbox.load_pairs(order[:args.n_queries], block) if args.kind != "identity" else []
    n_sel = max(20, args.n_queries // 5) if budget else 0
    train, sel = budget[:len(budget) - n_sel], budget[len(budget) - n_sel:]
    report = blackbox.load_pairs(order[-args.n_report:], block)
    print(f"loaded {len(train)} train + {len(sel)} selection (= N {args.n_queries}) / {len(report)} report queries "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    _, scale = feature_stats(report if not train else train, dev, 1)
    module = blackbox.build(args.kind).to(dev)
    info = {}
    if args.kind in ("linear", "mlp"):
        module.mu, module.sd = feature_stats(train, dev, 0)
        module.scale_out.copy_(scale)
    err0 = evaluate(module, report, scale, dev)  # every replacement starts as the identity
    print(f"identity: report {err0:.4f}", flush=True)
    if args.kind == "linear":
        info = fit_ridge(module, train, sel, scale, dev)
        fit_ridge(module, budget, None, scale, dev, lams=(info["lambda"],))  # refit on all N
    elif args.kind != "identity":
        info = fit_sgd(module, train, sel, scale, dev, args)
    err = evaluate(module, report, scale, dev)
    print(f"fitted: report {err:.4f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in module.parameters())
    blackbox.save(out, args.kind, block, module.cpu(), {"n_queries": args.n_queries})
    out.with_suffix(".json").write_text(json.dumps({
        "block": block, "kind": args.kind, "n_queries": args.n_queries, "n_train": len(train), "n_selection": len(sel),
        "n_report": len(report), "params": n_params, "report_identity": err0, "report_fitted": err,
        "minutes": (time.time() - t0) / 60, **info}, indent=2))


if __name__ == "__main__":
    main()
