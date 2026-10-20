"""
Expérience 3 — apprendre un contrôleur DANS le world model, puis le déployer sur le Franka.

Tâche : suivre des références articulaires rapides q_ref(t) (sommes de sinus, 0,3–1,2 Hz).
Les servos de position sont en retard et saturent (12 N·m aux poignets) : commander
a_t = q_ref(t+1) laisse une erreur de suivi.
Politique : a_t = q_ref(t+1) + 0,5 · tanh(MLP(ô_t − ref_t, q_ref(t+1..t+5) − q_ref(t))).
Entraînement : rétropropagation à travers le world model figé, sur H pas imaginés.
Évaluation : suivi sur MuJoCo (vérité terrain) pour 20 références inédites de 15 s.

Usage : python imagine.py --wm ren_H100_n128_s0 --models <dossier> --data franka.npz [--H 250]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from torch import nn

import franka_data as fd
from wm import NA, NO, Data, LatentWM, MambaWM, MLPStep

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
OUT = os.environ.get("OUT_DIR", "experiments/world_model/results")
LOOK = 5


def references(n, T, seed):
    """q_ref : (n, T + LOOK + 1, 7), dans 70 % de la plage articulaire."""
    m, _ = fd.load()
    lo, hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    rng = np.random.default_rng(seed)
    t = np.arange(T + LOOK + 1) * 0.02
    q = np.zeros((n, len(t), 7))
    for i in range(n):
        for j in range(7):
            for _ in range(2):
                f, ph = rng.uniform(0.3, 1.2), rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.1, 0.25) * half[j]
                q[i, :, j] += amp * np.sin(2 * np.pi * f * t + ph)
        q[i] += mid + rng.uniform(-0.3, 0.3, 7) * half
    return np.clip(q, lo + 0.05 * half, hi - 0.05 * half)


class Policy(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14 + 7 * LOOK, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, 7))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, dq, ref_now, ref_future):
        """Grandeurs physiques (rad, rad/s). Retourne la cible articulaire (7)."""
        feat = torch.cat([q - ref_now, 0.1 * dq, (ref_future - ref_now[:, None]).flatten(1)], -1)
        return ref_future[:, 0] + 0.5 * torch.tanh(self.net(feat))


def load_wm(path, name):
    kind = name.split("_H")[0]
    n = int(name.split("_n")[1].split("_")[0])
    model = MLPStep() if kind.startswith("mlp") else (MambaWM() if kind == "mamba2" else LatentWM(kind, n))
    model.load_state_dict(torch.load(path))
    for p in model.parameters():
        p.requires_grad_(False)
    return model, kind


def imagine(model, kind, data, policy, o0, ref):
    """Rollout imaginé. o0 : (B, 18) physique ; ref : (B, T + LOOK + 1, 7). Retourne q (B, T, 7)."""
    mo, so = torch.tensor(data.mo), torch.tensor(data.so)
    ma, sa = torch.tensor(data.ma), torch.tensor(data.sa)
    B, T = ref.shape[0], ref.shape[1] - LOOK - 1
    on = (o0 - mo) / so
    grip = torch.zeros(B, 1)
    qs = []
    if kind.startswith("mlp"):
        o = on
        for t in range(T):
            ophys = o * so + mo
            a = policy(ophys[:, :7], ophys[:, 9:16], ref[:, t], ref[:, t + 1:t + 1 + LOOK])
            an = (torch.cat([a, grip], -1) - ma) / sa
            o = o + 0.1 * model.f(torch.cat([o, an], -1))
            qs.append((o * so + mo)[:, :7])
    else:
        ctx = model.prepare()
        x = model.init_state(on)
        ophys = o0
        for t in range(T):
            a = policy(ophys[:, :7], ophys[:, 9:16], ref[:, t], ref[:, t + 1:t + 1 + LOOK])
            an = (torch.cat([a, grip], -1) - ma) / sa
            x, onext = model.advance(ctx, x, an)
            ophys = onext * so + mo
            qs.append(ophys[:, :7])
    return torch.stack(qs, 1)


def real_tracking(policy, ref):
    """Déploiement sur MuJoCo. ref : numpy (N, T + LOOK + 1, 7). Retourne RMSE (rad) par trajectoire."""
    m, d = fd.load()
    N, T = ref.shape[0], ref.shape[1] - LOOK - 1
    errs = []
    for i in range(N):
        import mujoco
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = ref[i, 0]
        d.qpos[7:] = 0.02
        mujoco.mj_forward(m, d)
        e = []
        for t in range(T):
            q = torch.tensor(d.qpos[:7], dtype=torch.float32)[None]
            dq = torch.tensor(d.qvel[:7], dtype=torch.float32)[None]
            r = torch.tensor(ref[i], dtype=torch.float32)
            with torch.no_grad():
                a = policy(q, dq, r[t][None], r[t + 1:t + 1 + LOOK][None])[0].numpy() \
                    if policy is not None else ref[i, t + 1]
            d.ctrl[:7] = a
            d.ctrl[7] = 0.0
            for _ in range(fd.SUBSTEPS):
                mujoco.mj_step(m, d)
            e.append(np.sum((d.qpos[:7] - ref[i, t + 1]) ** 2))
        errs.append(np.sqrt(np.mean(e) / 7))
    return np.array(errs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", default="franka.npz")
    ap.add_argument("--H", type=int, default=250)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    data = Data(a.data)
    path = glob.glob(f"{a.models}/**/{a.wm}.pt", recursive=True)[0]
    model, kind = load_wm(path, a.wm)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    ref_train = torch.tensor(references(512, a.H, 100 + a.seed), dtype=torch.float32)
    log = dict(loss=[], grad=[], nonfinite=0)
    t0 = time.time()
    for it in range(a.iters):
        idx = torch.randint(0, 512, (32,))
        ref = ref_train[idx]
        o0 = torch.cat([ref[:, 0], torch.full((32, 2), 0.02), torch.zeros(32, 9)], -1)
        q = imagine(model, kind, data, policy, o0, ref)
        loss = ((q - ref[:, 1:a.H + 1]) ** 2).mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            log["nonfinite"] += 1
            continue
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if torch.isfinite(g):
            opt.step()
        else:
            log["nonfinite"] += 1
        if it % 10 == 0:
            log["loss"].append(loss.item())
            log["grad"].append(g.item())
    log["time"] = time.time() - t0
    ref_test = references(20, 750, 999)
    with torch.no_grad():
        rt = torch.tensor(ref_test, dtype=torch.float32)
        o0 = torch.cat([rt[:, 0], torch.full((20, 2), 0.02), torch.zeros(20, 9)], -1)
        q_im = imagine(model, kind, data, policy, o0, rt)
        imag_rmse = float(((q_im - rt[:, 1:751]) ** 2).mean().sqrt())
    real = real_tracking(policy, ref_test)
    base = real_tracking(None, ref_test)
    res = dict(wm=a.wm, H=a.H, seed=a.seed, log=log, imagined_rmse=imag_rmse,
               real_rmse_mean=float(real.mean()), real_rmse=real.tolist(),
               baseline_rmse_mean=float(base.mean()), baseline_rmse=base.tolist())
    name = f"imagine_{a.wm}_H{a.H}_s{a.seed}"
    json.dump(res, open(f"{OUT}/{name}.json", "w"))
    print(name, f"{log['time']:.0f}s", "imagined", round(imag_rmse, 4), "real", round(real.mean(), 4),
          "baseline", round(base.mean(), 4), "nonfinite", log["nonfinite"], flush=True)
