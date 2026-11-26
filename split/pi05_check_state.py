"""P0/P1: π0.5 loads and answers, and its LIBERO policy ignores the robot state.

Same images, prompt and noise with two different states should give bitwise identical actions; a different image
should not (shows the check can see a difference). Runs in openpi's environment:
    vla_openpi python split/pi05_check_state.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_server import Server, load_policy


def main():
    policy = load_policy(os.path.join(os.environ["VLA_OPENPI"], "checkpoints/pi05_libero_pytorch"), "pi05_libero", "cuda:0")
    server = Server(policy, seed=7, recorder=None)
    rng = np.random.default_rng(0)
    image, wrist, other_image = (rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3))
    state = rng.normal(size=8)
    meta = {"task_id": 0, "episode": 0, "query": 0, "prompt": "put the black bowl on the plate"}

    t0 = time.perf_counter()
    base = server.infer(meta, image, wrist, state)
    first_s = time.perf_counter() - t0
    again = server.infer(meta, image, wrist, state)
    other_state = server.infer(meta, image, wrist, state + rng.normal(size=8))
    zero_state = server.infer(meta, image, wrist, np.zeros(8))
    other_img = server.infer(meta, other_image, wrist, state)
    other_noise = server.infer({**meta, "query": 1}, image, wrist, state)
    result = {
        "P0_actions_shape": list(base.shape),
        "first_query_s": first_s,
        "mean_query_ms": 1e3 * float(np.mean(server.times["compute"][1:])),
        "repeat_bitwise_equal": bool(np.array_equal(base, again)),
        "P1_other_state_bitwise_equal": bool(np.array_equal(base, other_state)),
        "P1_zero_state_bitwise_equal": bool(np.array_equal(base, zero_state)),
        "other_image_max_diff": float(np.abs(base - other_img).max()),
        "other_noise_max_diff": float(np.abs(base - other_noise).max()),
        **server.stats(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
