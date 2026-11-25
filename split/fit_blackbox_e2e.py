"""Black box, end to end (RESULTS §17): fine-tune a fit_blackbox.py replacement for layers a..b through the frozen
rest of the model, so its actions match the recorded ones. Loss: action L1 (normalized) + the reconstruction loss.

  cd openvla-oft && python ../split/fit_blackbox_e2e.py --suite libero_spatial --init .../L1-9_tf2_n5000.pt \
      --record_dir $VLA_DATA/blackbox/record9/libero_spatial --n_queries 5000 --out .../L1-9_tf2_n5000_e2e.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.robot_utils import get_model  # noqa: E402
import paths  # noqa: E402
import blackbox  # noqa: E402
from model2 import Model2  # noqa: E402
from secret_lora import query_files, run_layers  # noqa: E402
from verify_offline import CHECKPOINTS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--init", required=True, help="Reconstruction fit to start from")
    p.add_argument("--record_dir", required=True)
    p.add_argument("--n_queries", type=int, required=True)
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--optim", default="adamw", help="adamw, or adamw8bit (bitsandbytes; tf4 does not fit 24 GB otherwise)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    module, block, meta = blackbox.load(args.init, dev)
    a, b = block
    assert a == 1, "queries start from the recorded LLM input, so the block must start at layer 1"
    module.float().train()

    order = query_files(args.record_dir)
    random.Random(args.subset_seed).shuffle(order)  # same subset and split as fit_blackbox.py
    budget = order[:args.n_queries]
    n_sel = max(20, args.n_queries // 5)
    train_f, sel_f = budget[:-n_sel], budget[-n_sel:]

    cfg = rle.GenerateConfig(pretrained_checkpoint=CHECKPOINTS[args.suite], task_suite_name=args.suite)
    vla = get_model(cfg).eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    llama = vla.language_model.model
    del vla.vision_backbone, vla.projector  # not needed: queries start from the recorded LLM input
    llama.embed_tokens.to("cpu")  # not needed either (inputs are embeddings); nor the LM head
    vla.language_model.lm_head.to("cpu")
    for i in range(b):
        llama.layers[i].to("cpu")  # secret layers: the attacker does not have them
    torch.cuda.empty_cache()
    head = Model2.load(paths.SPLIT / args.suite / "model2", dev)
    for p in head.head.parameters():
        p.requires_grad_(False)
    low = torch.tensor(head.action_low, device=dev, dtype=torch.float32)
    high = torch.tensor(head.action_high, device=dev, dtype=torch.float32)

    def load(f):
        d = torch.load(f, weights_only=False)
        act = torch.tensor(d["actions"], dtype=torch.float32)
        return {"emb": d["inputs_embeds"][0], "mask": d["attention_mask"][0], "h": d["h"][b][0], "act": act}

    t0 = time.time()
    train = [load(f) for f in train_f]
    sel = [load(f) for f in sel_f]
    print(f"loaded {len(train)} train + {len(sel)} selection (= N {args.n_queries}) ({(time.time() - t0) / 60:.1f} min)",
          flush=True)
    scale = torch.cat([q["h"].float() for q in train[:500]]).std(0).clamp_min(1e-3).to(dev)

    def target(act):
        """Executed actions -> the head's normalized output."""
        act = act.to(dev)
        norm = 2 * (act - low) / (high - low + 1e-8) - 1
        grip = (act[:, -1] < 0).float()  # executed -1 (LIBERO) <- head output > 0.5
        norm[:, -1] = grip
        return norm

    def forward(q):
        emb, m = q["emb"].to(dev)[None], q["mask"].to(dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_b = module(emb)
            h = run_layers(vla, h_b, m, b + 1, llama.config.num_hidden_layers)
            h = llama.norm(h)
            T = h.shape[1]
            act = head.head.predict_action(h[:, T - 57:T - 1]).reshape(8, 7).float()
        return h_b[0], act

    @torch.no_grad()
    def evaluate(qs):
        module.eval()
        e = float(np.mean([float((forward(q)[1] - target(q["act"])).abs().mean()) for q in qs]))
        module.train()
        return e

    params = [p for p in module.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(s / args.steps * np.pi)))
    rng = random.Random(1)
    snapshot = lambda: {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}  # noqa: E731
    best = (evaluate(sel), 0, snapshot())
    log = [{"step": 0, "sel_action_l1": best[0]}]
    print(f"start: selection action L1 {best[0]:.4f}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.batch):
            q = rng.choice(train)
            h_b, act = forward(q)
            loss = (act - target(q["act"])).abs().mean()
            loss = loss + args.recon_weight * (((h_b.float() - q["h"].to(dev).float()) / scale) ** 2).mean()
            (loss / args.batch).backward()
            tot += float(loss) / args.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            e = evaluate(sel)
            log.append({"step": step, "train": tot, "sel_action_l1": e})
            print(f"step {step} train {tot:.4f} selection action L1 {e:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if e < best[0]:
                best = (e, step, snapshot())
    module.load_state_dict(best[2])
    module.cpu()
    blackbox.save(args.out, meta["kind"], block, module, {"n_queries": args.n_queries, "init": str(args.init)})
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "block": block, "kind": meta["kind"], "init": str(args.init), "n_queries": args.n_queries, "n_train": len(train),
        "n_selection": len(sel), "steps": args.steps, "lr": args.lr, "optim": args.optim, "recon_weight": args.recon_weight,
        "best_step": best[1], "sel_action_l1": best[0], "log": log}, indent=2))


if __name__ == "__main__":
    main()
