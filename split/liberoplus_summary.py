"""LIBERO-Plus summary: each readout vs. its real model on identical perturbed tasks, per suite and per kind of shift.

Checks that paired runs saw the same first image and instruction; paired difference with exact McNemar test and 95% CI
(as pi05_cv_summary.py). Writes $VLA_DATA/libero_plus/summary.json.
    python liberoplus_summary.py [results root, default $VLA_DATA/libero_plus]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from pi05_cv_summary import mcnemar_p

ROOT = paths.DATA / "libero_plus"
CLASSIFICATION = paths.REPO / "LIBERO-plus/libero/libero/benchmark/task_classification.json"
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]
PAIRS = [("oft_linear", "oft_real"), ("pi05_linear", "pi05_real"), ("pi05_mlp", "pi05_real")]
SETUPS = ["oft_real", "oft_linear", "pi05_real", "pi05_linear", "pi05_mlp"]
CATEGORIES = ["Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions",
              "Background Textures", "Sensor Noise", "Objects Layout"]


def load_setup(setup):
    """{suite/task<id>_ep0: success}, first-image hashes, prompts for one setup, from all its chunk files."""
    succ, md5, prompts = {}, {}, {}
    for f in sorted((ROOT / "evals" / setup).glob("*/*_eval_seed7_1ep_tasks*.json")):
        r = json.loads(f.read_text())
        suite = r["suite"]
        succ.update({f"{suite}/{k}": v for k, v in r["successes"].items()})
        md5.update({f"{suite}/{k}": v for k, v in r["first_image_md5"].items()})
        prompts.update({f"{suite}/task{k}": v for k, v in r["prompts"].items()})
    return succ, md5, prompts


def paired(a, b, keys):
    d = [int(a[k]) - int(b[k]) for k in keys]
    n = len(d)
    if n == 0:
        return None
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    only_a = sum(1 for k in keys if b[k] and not a[k])  # readout fails, real succeeds
    only_b = sum(1 for k in keys if a[k] and not b[k])
    return {"episodes": n, "readout": sum(a[k] for k in keys), "real": sum(b[k] for k in keys),
            "diff_pp": 100 * mean, "ci95_pp": [100 * (mean - 1.96 * sd / n**0.5), 100 * (mean + 1.96 * sd / n**0.5)],
            "only_readout_fails": only_a, "only_real_fails": only_b, "mcnemar_p": mcnemar_p(only_a, only_b)}


def main():
    global ROOT
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    cls = json.loads(CLASSIFICATION.read_text())
    category = {f"{s}/task{t['id'] - 1}_ep0": t["category"] for s, _ in SUITES for t in cls[s]}
    data = {s: load_setup(s) for s in SETUPS}
    out = {"success": {}, "pairs": {}, "checks": {}}

    for setup in SETUPS:
        succ = data[setup][0]
        if not succ:
            continue
        row = {}
        for cat in CATEGORIES + ["All"]:
            ks = [k for k in succ if cat == "All" or category[k] == cat]
            row[cat] = {"episodes": len(ks), "successes": sum(succ[k] for k in ks),
                        "rate": sum(succ[k] for k in ks) / len(ks) if ks else None}
        out["success"][setup] = row

    for a, b in PAIRS:
        (sa, ma, pa), (sb, mb, pb) = data[a], data[b]
        keys = sorted(set(sa) & set(sb))
        if not keys:
            continue
        out["checks"][f"{a} vs {b}"] = {
            "paired_episodes": len(keys),
            "same_first_image": sum(ma[k] == mb[k] for k in keys),
            "same_prompt": sum(pa[t] == pb[t] for t in set(pa) & set(pb)),
            "prompts_compared": len(set(pa) & set(pb)),
        }
        table = {}
        for suite, label in SUITES + [(None, "All suites")]:
            for cat in CATEGORIES + ["All"]:
                ks = [k for k in keys if (suite is None or k.startswith(suite + "/")) and (cat == "All" or category[k] == cat)]
                table[f"{label} | {cat}"] = paired(sa, sb, ks)
        out["pairs"][f"{a} vs {b}"] = table

    (ROOT / "summary.json").write_text(json.dumps(out, indent=2))
    for setup, row in out["success"].items():
        print(f"{setup:12s} " + "  ".join(f"{c.split()[0][:6]} {100 * v['rate']:.1f}" for c, v in row.items() if v["rate"] is not None)
              + f"  ({row['All']['episodes']} episodes)")
    for name, table in out["pairs"].items():
        print(f"\n{name}: {json.dumps(out['checks'][name])}")
        for key, r in table.items():
            if r and key.startswith("All suites"):
                print(f"  {key:36s} {r['readout']}/{r['episodes']} vs {r['real']}/{r['episodes']}  "
                      f"{r['diff_pp']:+.1f} pp [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]  p={r['mcnemar_p']:.3f}")
    print(f"wrote {ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
