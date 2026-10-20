"""
Données de l'expérience 2 : Franka Panda (MuJoCo Menagerie), actionneurs en position
(servos PD avec limites de couple), 50 Hz.

Action  a_t ∈ R⁸ : cibles articulaires (7) + pince (0..255).
Observation o_t ∈ R¹⁸ : qpos (9) + qvel (9).
Cibles : points aléatoires dans les butées, tenus 0,2 à 1,2 s, filtrés par un premier ordre
de constante de temps aléatoire (0 à 0,4 s) : mouvements lents et mouvements saturés.
"""
import os
import sys

import mujoco
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda it, **k: it  # noqa: E731

SUBSTEPS = 10            # 0,002 s × 10 = 0,02 s
MENAGERIE = os.environ.get("MENAGERIE", os.path.expanduser("~/mujoco_menagerie"))


def load():
    m = mujoco.MjModel.from_xml_path(os.path.join(MENAGERIE, "franka_emika_panda", "scene.xml"))
    return m, mujoco.MjData(m)


def target_sequence(rng, T, lo, hi):
    a = np.zeros((T, 8))
    t, cur = 0, rng.uniform(lo, hi)
    tau = rng.uniform(0.0, 0.4)
    filt = cur.copy()
    while t < T:
        hold = int(rng.uniform(0.2, 1.2) / 0.02)
        cur = rng.uniform(lo, hi)
        for _ in range(hold):
            if t >= T:
                break
            k = 1.0 if tau < 0.02 else 0.02 / tau
            filt = filt + k * (cur - filt)
            a[t] = filt
            t += 1
    return a


def rollout(m, d, actions, q0):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    obs = np.zeros((len(actions) + 1, 18))
    obs[0] = np.concatenate([d.qpos, d.qvel])
    for t, a in enumerate(actions):
        d.ctrl[:] = a
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        obs[t + 1] = np.concatenate([d.qpos, d.qvel])
    return obs


def generate(n, T, seed):
    m, d = load()
    rng = np.random.default_rng(seed)
    lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    lo, hi = mid - 0.85 * half, mid + 0.85 * half
    O, A = np.zeros((n, T + 1, 18), np.float32), np.zeros((n, T, 8), np.float32)
    for i in tqdm(range(n), desc=f"simulation {n}×{T}", mininterval=5, ncols=90):
        A[i] = target_sequence(rng, T, lo, hi)
        q0 = np.zeros(9)
        q0[:7] = A[i, 0, :7] + 0.1 * rng.normal(size=7)
        q0[:7] = np.clip(q0[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        q0[7:] = 0.02
        O[i] = rollout(m, d, A[i], q0)
    return O, A


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "franka.npz"
    Otr, Atr = generate(1000, 250, 0)      # 5 s
    Ova, Ava = generate(100, 250, 1)
    Ote, Ate = generate(50, 1500, 2)       # 30 s
    np.savez_compressed(out, Otr=Otr, Atr=Atr, Ova=Ova, Ava=Ava, Ote=Ote, Ate=Ate)
    print("saved", out, Otr.shape, Ote.shape, "finite:", np.isfinite(Otr).all(), np.isfinite(Ote).all())
