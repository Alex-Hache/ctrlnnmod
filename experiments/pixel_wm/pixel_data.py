"""
M3a — données image du Franka (MuJoCo, rendu hors écran OSMesa ou EGL).

Trajectoires comme dans experiments/world_model (servos de position, 50 Hz). On rend deux images
64×64 RGB (t, t+1) aux instants de départ des fenêtres d'entraînement, et à t = 0 pour le test.
Les images ne dépendent que de qpos : on rejoue qpos enregistré, sans resimuler.

Rendu : sans ombres, reflets ni ciel (OSMesa : 390 ms → 44 ms par image), en parallèle sur
tous les cœurs (un moteur de rendu par processus).
"""
import multiprocessing as mp
import os
import sys

import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "world_model"))
import franka_data as fd  # noqa: E402

RES = 64
STARTS = list(range(0, 201, 25))          # 9 départs de fenêtre (fenêtres ≤ 50 pas après 200)


def renderer(m):
    m.vis.quality.offsamples = 0
    m.vis.quality.shadowsize = 0
    r = mujoco.Renderer(m, RES, RES)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 135.0, -25.0
    return r, cam


_OFF = (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX)


def render_pair(m, d, r, cam, qpos_a, qpos_b):
    out = []
    for q in (qpos_a, qpos_b):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        for f in _OFF:
            r.scene.flags[f] = 0
        out.append(r.render().transpose(2, 0, 1))
    return np.stack(out)                                     # (2, 3, 64, 64) uint8


_W = {}


def _init():
    m, d = fd.load()
    r, cam = renderer(m)
    _W.update(m=m, d=d, r=r, cam=cam)


def _job(args):
    """Rend les paires d'une trajectoire : qpos (T+1, 9), départs."""
    Q, starts = args
    w = _W
    return np.stack([render_pair(w["m"], w["d"], w["r"], w["cam"], Q[s], Q[s + 1]) for s in starts])


def render_all(O, starts, pool, desc):
    jobs = [(o[:, :9], starts) for o in O]
    it = pool.imap(_job, jobs, chunksize=8)
    return np.stack(list(tqdm(it, total=len(jobs), desc=desc, mininterval=5, ncols=90)))


def build(n_tr, n_te, T_tr=250, T_te=1500):
    Otr, Atr = fd.generate(n_tr, T_tr, 0)
    Ova, Ava = fd.generate(max(n_tr // 10, 20), T_tr, 1)
    Ote, Ate = fd.generate(n_te, T_te, 2)
    nproc = int(os.environ.get("NPROC", os.cpu_count() or 1))
    print(f"rendu sur {nproc} processus", flush=True)
    with mp.get_context("fork").Pool(nproc, initializer=_init) as pool:
        Itr = render_all(Otr, STARTS, pool, "rendu train")
        Iva = render_all(Ova, STARTS, pool, "rendu val")
        Ite = render_all(Ote, [0], pool, "rendu test")[:, 0]
    return dict(Otr=Otr, Atr=Atr, Itr=Itr, Ova=Ova, Ava=Ava, Iva=Iva, Ote=Ote, Ate=Ate, Ite=Ite,
                starts=np.array(STARTS))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "franka_pix.npz"
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    z = build(n_tr, 50)
    np.savez(out, **z)
    print("saved", out, {k: v.shape for k, v in z.items()}, "pixel mean", z["Itr"].mean())
