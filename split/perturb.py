"""Deterministic distribution-shift perturbations for the robustness test.

Seeded per episode; never uses numpy's global RNG, which LIBERO uses for fixture placement."""

import zlib

import numpy as np

KINDS = ("none", "noise", "dark", "camera", "objects")
NOISE_SIGMA = 20.0
DARK = 0.3
CAMERA_SHIFT = np.array([0.0, 0.06, 0.04])
OBJECT_SHIFT = 0.03


def _seed(data):
    return zlib.crc32(np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray) else str(data).encode())


def perturb_env(env, kind):
    """Apply the perturbation after every set_init_state (LIBERO rebuilds the model on reset)."""
    if kind not in ("dark", "camera", "objects"):
        return env
    set_init_state = env.set_init_state

    def perturbed_set_init_state(init_state):
        set_init_state(init_state)
        sim, model = env.sim, env.sim.model
        if kind == "dark":
            model.light_diffuse[:] = DARK * model.light_diffuse
        elif kind == "camera":
            model.cam_pos[model.camera_name2id("agentview")] += CAMERA_SHIFT
        elif kind == "objects":
            rng = np.random.default_rng(_seed(np.asarray(init_state)))
            start = sim.data.qpos.copy()
            free = [model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == 0]  # movable objects
            allowed = _penetrations(sim)
            for _ in range(50):
                qpos = start.copy()
                for adr in free:
                    qpos[adr:adr + 2] += rng.uniform(-OBJECT_SHIFT, OBJECT_SHIFT, 2)
                sim.data.qpos[:] = qpos
                sim.forward()
                if _penetrations(sim) <= allowed:
                    break
            else:
                sim.data.qpos[:] = start
                sim.forward()
        return env.regenerate_obs_from_state(env.get_sim_state())

    env.set_init_state = perturbed_set_init_state
    return env


def _penetrations(sim, depth=0.002):
    """Number of contacts deeper than `depth` metres."""
    return sum(sim.data.contact[i].dist < -depth for i in range(sim.data.ncon))


def episode_rng(episode_key):
    return np.random.default_rng(_seed(episode_key))


def add_noise(image, rng):
    noisy = image.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
