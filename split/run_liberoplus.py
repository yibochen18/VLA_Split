"""LIBERO-Plus: real models and their clean-LIBERO readouts (fold 0, which never saw init state 0) on every perturbed
task, one episode each.

    oft_real / oft_linear     OpenVLA-OFT cut-A split with its MLP head / the §8 linear map
    pi05_real / pi05_linear   pi05_libero / the §15 linear map in place of the action expert
    pi05_mlp                  the §15 MLP in place of the action expert
Each task builds a fresh env, so all setups see identical scenes (checked via each episode's first-image hash).

Jobs are 250-task chunks per (setup, suite); finished chunks are skipped, so rerunning resumes. A GPU holds two π0.5
jobs (~8 GB each) or one OFT job (~16 GB). Keep at most two simulators rendering per GPU: more made MuJoCo's EGL
rendering fail and camera images change.
    source env.sh && vla_activate_plus && python split/run_liberoplus.py [--gpus 1 | --gpu_units 1:2 0:1] [--dry_run]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

SUITES = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}
SETUPS = ["pi05_real", "pi05_linear", "pi05_mlp", "oft_real", "oft_linear"]
UNITS = {"pi05": 1, "oft": 2}  # GPU share: 2 units per GPU


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="+", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(","))
    p.add_argument("--setups", nargs="+", default=SETUPS, choices=SETUPS)
    p.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--task_limit", type=int, default=None, help="Only the first N tasks per suite (smoke tests)")
    p.add_argument("--tasks", type=int, nargs="+", default=None, help="Only these task ids, one job each (smoke tests)")
    p.add_argument("--root", default=str(paths.DATA / "libero_plus" / "evals"))
    p.add_argument("--gpu_units", nargs="+", default=None,
                   help="GPU:units, e.g. 1:2 0:1 (default 2 per GPU); a π0.5 job takes 1 unit, an OFT job 2")
    p.add_argument("--oft_first", action="store_true", help="Queue all OFT jobs before all π0.5 jobs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def head_dir(setup, suite):
    d = paths.DATA
    return {"oft_real": None, "oft_linear": d / "linear_head" / "heads" / f"{suite}_fold0",
            "pi05_real": None, "pi05_linear": d / "pi05" / "cv" / "heads" / f"{suite}_fold0",
            "pi05_mlp": Path(os.environ.get("PI05_MLP_CV", d / "pi05" / "mlp_cv_std")) / "heads" / f"{suite}_fold0"}[setup]


def job(setup, suite, a, b, root):
    """(kind, command, result path, log path) for tasks a..b-1."""
    out = Path(root) / setup
    head = head_dir(setup, suite)
    common = ["--suite", suite, "--episodes_per_task", "1", "--task_from", str(a), "--task_to", str(b),
              "--plus_prompt", "--tag", setup]
    if setup.startswith("pi05"):
        cmd = ["python", "../split/run_pi05_eval.py", *common, "--out_root", str(out),
               "--socket", f"/tmp/vla_lp_{setup}_{suite}_{a}.sock"]
        cmd += {"pi05_linear": ["--linear", str(head)], "pi05_mlp": ["--mlp", str(head)]}.get(setup, [])
        result = out / suite / f"pi05_eval_seed7_1ep_tasks{a}-{b}_{setup}.json"
        kind = "pi05"
    else:
        cmd = ["python", "../split/run_split_eval.py", *common, "--split_root", str(out),
               "--socket", f"/tmp/vla_lp_{setup}_{suite}_{a}.sock"]
        if head:
            cmd += ["--bundle", str(head)]
        else:
            cmd += ["--bundle", str(paths.SPLIT / suite / "model2")]
        result = out / suite / f"split_eval_seed7_1ep_tasks{a}-{b}_{setup}.json"
        kind = "oft"
    return kind, cmd, result, result.with_suffix(".console.log")


def build_jobs(args):
    jobs = []
    if args.oft_first:  # same jobs, other order
        by_kind = {"oft": [], "pi05": []}
        for j in build_jobs(argparse.Namespace(**{**vars(args), "oft_first": False})):
            by_kind[j[4]].append(j)
        return by_kind["oft"] + by_kind["pi05"]
    for suite in args.suites:
        n = SUITES[suite] if args.task_limit is None else min(args.task_limit, SUITES[suite])
        ranges = [(t, t + 1) for t in args.tasks] if args.tasks else [(a, min(a + args.chunk, n)) for a in range(0, n, args.chunk)]
        for kind in ("pi05", "oft"):
            for a, b in ranges:
                for setup in args.setups:
                    if setup.startswith(kind):
                        jobs.append((setup, suite, a, b, *job(setup, suite, a, b, args.root)))
    return jobs


def log(msg):
    print(f"=== [{time.strftime('%F %T')}] {msg}", flush=True)


def main():
    args = parse_args()
    jobs = build_jobs(args)
    missing = {(setup, suite) for setup, suite, *_ in jobs
               if head_dir(setup, suite) is not None and not (head_dir(setup, suite) / "config.json").exists()}
    for setup, suite in sorted(missing):
        log(f"WARNING: no readout {head_dir(setup, suite)}; skipping {setup} on {suite}")
    jobs = [j for j in jobs if (j[0], j[1]) not in missing]
    pending = [j for j in jobs if not j[6].exists()]
    log(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run on GPUs "
        f"{args.gpu_units or args.gpus}")
    if args.dry_run:
        for j in pending[:10]:
            print(j[0], j[1], j[2], j[3], " ".join(j[5]))
        return

    cap = {g: 2 for g in args.gpus}
    if args.gpu_units:
        cap = {g: int(u) for g, u in (x.split(":") for x in args.gpu_units)}
    free = dict(cap)
    running, failed = [], []
    # Adopt chunks still running from an earlier orchestrator, so none runs twice
    for j in list(pending):
        pat = f"--suite {j[1]} --episodes_per_task 1 --task_from {j[2]} --task_to {j[3]} .*--tag {j[0]}( |$)"
        out = subprocess.run(["pgrep", "-f", "--", pat], capture_output=True, text=True).stdout.split()
        pids = [int(x) for x in out if int(x) != os.getpid()]
        if pids:
            g = (open(f"/proc/{pids[0]}/environ").read().split("CUDA_VISIBLE_DEVICES=")[1].split("\0")[0]
                 if "CUDA_VISIBLE_DEVICES=" in open(f"/proc/{pids[0]}/environ").read() else args.gpus[0])
            pending.remove(j)
            free[g] -= UNITS[j[4]]
            running.append((pids[0], g, UNITS[j[4]], j))
            log(f"adopted running {j[0]} {j[1]} tasks {j[2]}-{j[3]} (pid {pids[0]}, GPU {g})")
    oft_dir = paths.REPO / "openvla-oft"
    while pending or running:
        for r in running[:]:
            proc, g, units, j = r
            if isinstance(proc, int):  # adopted: not our child, so poll the pid
                finished, code = not Path(f"/proc/{proc}").exists(), None
            else:
                finished, code = proc.poll() is not None, proc.returncode
            if finished:
                running.remove(r)
                free[g] += units
                ok = code in (0, None) and j[6].exists()
                log(f"{'done' if ok else 'FAILED'} {j[0]} {j[1]} tasks {j[2]}-{j[3]} (exit {code})")
                if not ok:
                    failed.append(j)
        # In order, except that π0.5 jobs may pass a waiting OFT job on GPUs too small for OFT
        oft_blocked = False
        for j in list(pending):
            units = UNITS[j[4]]
            allowed = [g for g in cap if not oft_blocked or cap[g] < UNITS["oft"]]
            g = next((g for g in allowed if free[g] >= units), None)
            if g is None:
                if j[4] == "oft":
                    oft_blocked = True
                    continue
                break
            pending.remove(j)
            free[g] -= units
            j[6].parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            proc = subprocess.Popen(j[5], cwd=oft_dir, env=env, stdout=open(j[7], "w"), stderr=subprocess.STDOUT)
            running.append((proc, g, units, j))
            log(f"start {j[0]} {j[1]} tasks {j[2]}-{j[3]} on GPU {g}")
        time.sleep(5)
    log(f"ALL DONE, {len(failed)} failed" + "".join(f"\n  {j[0]} {j[1]} {j[2]}-{j[3]}: {j[7]}" for j in failed))


if __name__ == "__main__":
    main()
