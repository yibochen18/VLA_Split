"""Paths and environment settings from env.sh, for Python code."""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNTIME_VARS = ("HF_HOME", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL")


def _load_env():
    if "VLA_DATA" in os.environ:
        return dict(os.environ)
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO / "env.sh"}" && env -0'], capture_output=True, check=True
    ).stdout.decode()
    env = dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
    for key in RUNTIME_VARS:
        os.environ.setdefault(key, env[key])
    return env


_env = _load_env()
DATA = Path(_env["VLA_DATA"])
LOGS = Path(_env["VLA_LOGS"])
REFERENCE = Path(_env["VLA_REFERENCE"])
SPLIT = Path(_env["VLA_SPLIT"])
EXTRACTION = Path(_env["VLA_EXTRACTION"])
