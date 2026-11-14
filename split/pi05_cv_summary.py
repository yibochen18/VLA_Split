"""π0.5 cross-validation summary: a readout (linear map or MLP) vs. π0.5 on identical scenes, paired per episode
(as RESULTS.md §8).

Fold f: the readout run from start 10f (run_pi05_cv.sh / run_pi05_mlp_cv.sh) against π0.5 run the same way (fold 0:
the baseline run, which also starts at 0). Exact McNemar test on episodes where exactly one fails; 95% CI of the
difference from a normal approximation on the paired differences.
    python pi05_cv_summary.py            linear map -> $VLA_DATA/pi05/cv/summary.json
    python pi05_cv_summary.py mlp [dir]  MLP        -> $VLA_DATA/pi05/mlp_cv_std/summary.json
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

ROOT = paths.DATA / "pi05"
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]


def load(p):
    return json.loads(Path(p).read_text())


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(lin, ref):
    keys = sorted(set(lin) & set(ref))
    d = [int(lin[k]) - int(ref[k]) for k in keys]
    n, mean = len(d), sum(d) / len(d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n)
    only_lin_fails = sum(1 for k in keys if ref[k] and not lin[k])
    only_ref_fails = sum(1 for k in keys if lin[k] and not ref[k])
    return {"episodes": n, "linear": sum(lin[k] for k in keys), "pi05": sum(ref[k] for k in keys),
            "only_linear_fails": only_lin_fails, "only_pi05_fails": only_ref_fails,
            "diff_pp": 100 * mean, "ci95_pp": [100 * (mean - half), 100 * (mean + half)],
            "mcnemar_p": mcnemar_p(only_lin_fails, only_ref_fails)}


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "linear"
    assert kind in ("linear", "mlp"), kind
    cv = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ("cv" if kind == "linear" else "mlp_cv_std")
    rows, all_lin, all_ref = [], {}, {}
    for suite, label in SUITES:
        lin, ref, folds = {}, {}, []
        for f in range(5):
            lp = cv / f"evals/{suite}/pi05_eval_seed7_10ep_from{10 * f}_{kind}_fold{f}.json" if f else \
                cv / f"evals/{suite}/pi05_eval_seed7_10ep_{kind}_fold0.json"
            rp = ROOT / f"cv/evals/{suite}/pi05_eval_seed7_10ep_from{10 * f}_ref_fold{f}.json" if f else \
                ROOT / f"evals/{suite}/pi05_eval_seed7_50ep.json"
            if not (lp.exists() and rp.exists()):
                continue
            l_s = load(lp)["successes"]
            r_s = {k: v for k, v in load(rp)["successes"].items() if 10 * f <= int(k.split("_ep")[1]) < 10 * f + 10}
            head = load(cv / f"heads/{suite}_fold{f}/config.json")
            folds.append({"fold": f, "linear": sum(l_s.values()), "pi05": sum(r_s.values()),
                          "test_l1": head["metrics"]["test"]["l1_mean"],
                          "lambda" if kind == "linear" else "steps": head["lambda" if kind == "linear" else "steps"],
                          "train_queries": head["num_train_queries"]})
            lin.update({f"{suite}/{k}": v for k, v in l_s.items()})
            ref.update({f"{suite}/{k}": v for k, v in r_s.items()})
        if not folds:
            continue
        rows.append({"suite": label, "folds": folds, **paired(lin, ref),
                     "test_l1_mean": sum(x["test_l1"] for x in folds) / len(folds)})
        all_lin.update(lin)
        all_ref.update(ref)
    if all_lin:
        rows.append({"suite": "All", **paired(all_lin, all_ref)})
    for r in rows:  # fields say "linear" for either readout
        r["readout"] = kind
    out = cv / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['suite']:8s} {kind} {r['linear']}/{r['episodes']}  pi05 {r['pi05']}/{r['episodes']}  "
              f"diff {r['diff_pp']:+.1f} pp [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]  "
              f"only-{kind}-fails {r['only_linear_fails']} / only-pi05-fails {r['only_pi05_fails']}  p={r['mcnemar_p']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
