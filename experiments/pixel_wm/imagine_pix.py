"""
M3a, étape 2 — apprendre un contrôleur DANS un world model sur images, puis le déployer sur MuJoCo.

Même tâche et même politique que l'expérience 3 (experiments/world_model/imagine.py) :
suivre des références articulaires rapides. Différence : l'état initial imaginé vient de deux
images 64×64 (encodeur CNN), puis la dynamique latente avance seule, pilotée par la politique.
Le world model a été entraîné sur H = 50 pas ; l'imagination dure H = 250 pas (extrapolation).

Usage : python imagine_pix.py --wm pix_sren_n1024_H50_s0 [--H 250] [--seed 0]
Poids lus dans /data/ckpt/<wm>.pt, normalisation lue dans /data/franka_pix.npz.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from tqdm import tqdm

# ordre voulu : torch / triton avant MuJoCo + OSMesa (sinon erreur de segmentation à l'import de triton)
from pwm import NA, NO, dev, make_model  # noqa: I001
from imagine import LOOK, Policy, real_tracking, references
import pixel_data as pdm

OUT = os.environ.get("OUT_DIR", "experiments/pixel_wm/results")


def _render(q7):
    w = pdm._W
    q = np.concatenate([q7, [0.02, 0.02]])
    return pdm.render_pair(w["m"], w["d"], w["r"], w["cam"], q, q)        # au repos : 2 images égales


def initial_images(ref):
    """ref : (N, T, 7) numpy → images (N, 2, 3, 64, 64) uint8 de la pose initiale."""
    nproc = int(os.environ.get("NPROC", "4"))
    with mp.get_context("fork").Pool(nproc, initializer=pdm._init) as pool:
        return np.stack(pool.map(_render, list(ref[:, 0]), chunksize=8))


class Latent:
    """Interface pas à pas du modèle figé : prepare / init_state / advance → (état, ô normalisé).
    Les références (baselines.py) ont déjà init_state / advance ; ô peut être (K, B, 18)."""

    def __init__(self, m):
        self.m = m
        self.native = hasattr(m, "advance")

    @torch.no_grad()
    def prepare(self):
        return self.m.cell.frame() if getattr(self.m, "kind", None) in ("ren", "sren") else None

    def init_state(self, img):
        m = self.m
        if self.native:
            return m.init_state(img)
        x0 = m.enc(img)
        if m.kind == "gru":
            return x0[None].contiguous()
        if m.kind in ("ren", "sren"):
            return x0
        sts, tok = [], x0
        for b in m.blocks:
            st = tuple(s.to(dev) for s in b.init_state(img.shape[0]))
            st, tok = b.step(st, tok)
            sts.append(st)
        return sts

    def advance(self, fr, x, an):
        m = self.m
        if self.native:
            return m.advance(x, an)
        if m.kind == "gru":
            out, h = m.cell(an[:, None], x)
            return h, m.dec(out[:, 0])
        if m.kind == "ren":
            xp, _ = m.cell.step(fr, x, an)
            return xp, m.dec(xp)
        if m.kind == "sren":
            xn = m.cell.step(fr, x, an)
            return xn, m.dec(xn)
        tok, new = m.act_emb(an), []
        for b, st in zip(m.blocks, x):
            st, tok = b.step(st, tok)
            new.append(st)
        return new, m.dec(tok)


def imagine(lat, norm, policy, img, ref, fr):
    """img : (B, 2, 3, 64, 64) ; ref : (B, T + LOOK + 1, 7) physique. Retourne q imaginé (B, T, 7)."""
    mo, so, ma, sa = norm
    B, T = ref.shape[0], ref.shape[1] - LOOK - 1
    grip = torch.zeros(B, 1, device=dev)
    x = lat.init_state(img)
    q, dq = ref[:, 0], torch.zeros(B, 7, device=dev)
    qs = []
    for t in range(T):
        a = policy(q, dq, ref[:, t], ref[:, t + 1:t + 1 + LOOK])
        an = (torch.cat([a, grip], -1) - ma) / sa
        x, on = lat.advance(fr, x, an)
        o = on * so + mo                       # (B, 18), ou (K, B, 18) pour un ensemble
        qs.append(o[..., :7])
        q, dq = o[..., :7], o[..., 9:16]
        if o.dim() == 3:                       # ensemble : la politique voit la moyenne
            q, dq = q.mean(0), dq.mean(0)
    return torch.stack(qs, -2)                 # (B, T, 7) ou (K, B, T, 7) : coût moyen sur les membres


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--H", type=int, default=250)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/data/franka_pix.npz")
    ap.add_argument("--ckpt", default="/data/ckpt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    z = np.load(a.data)
    O, A = z["Otr"].reshape(-1, NO), z["Atr"].reshape(-1, NA)
    norm = [torch.tensor(v, dtype=torch.float32, device=dev)
            for v in (O.mean(0), O.std(0) + 1e-3, A.mean(0), A.std(0) + 1e-3)]   # comme pwm.Data
    kind, n = a.wm.split("_")[1], int(a.wm.split("_n")[1].split("_")[0])
    model = make_model(kind, n).to(dev)
    model.load_state_dict(torch.load(f"{a.ckpt}/{a.wm}.pt", map_location=dev))
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()                       # cuDNN : la passe arrière du GRU exige le mode train
    lat = Latent(model)
    fr = lat.prepare()

    ref_np = references(512, a.H, 100 + a.seed)
    ref_te_np = references(20, 750, 999)
    print("rendu des poses initiales", flush=True)
    img_tr = torch.tensor(initial_images(ref_np), device=dev)
    img_te = torch.tensor(initial_images(ref_te_np), device=dev)
    ref_tr = torch.tensor(ref_np, dtype=torch.float32, device=dev)

    policy = Policy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    log = dict(loss=[], grad=[], nonfinite=0)
    t0 = time.time()
    bar = tqdm(range(a.iters), desc=f"imagination {a.wm} H={a.H}", mininterval=10, ncols=110)
    for it in bar:
        idx = torch.randint(0, 512, (32,), device=dev)
        q = imagine(lat, norm, policy, img_tr[idx], ref_tr[idx], fr)
        loss = ((q - ref_tr[idx, 1:a.H + 1]) ** 2).mean()
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
            bar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{g.item():.2e}")
    log["time"] = time.time() - t0

    with torch.no_grad():
        rt = torch.tensor(ref_te_np, dtype=torch.float32, device=dev)
        q_im = imagine(lat, norm, policy, img_te, rt, fr)
        imag_rmse = float(((q_im - rt[:, 1:751]) ** 2).mean().sqrt())
    pol_cpu = policy.cpu()
    print("évaluation sur MuJoCo", flush=True)
    real = real_tracking(pol_cpu, ref_te_np)
    base = real_tracking(None, ref_te_np)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    res = dict(wm=a.wm, H=a.H, seed=a.seed, gpu=gpu, log=log, imagined_rmse=imag_rmse,
               real_rmse_mean=float(real.mean()), real_rmse=real.tolist(),
               baseline_rmse_mean=float(base.mean()), baseline_rmse=base.tolist())
    name = f"imagine_{a.wm}_H{a.H}_s{a.seed}"
    json.dump(res, open(f"{OUT}/{name}.json", "w"))
    print(name, f"{log['time']:.0f}s", "imagined", round(imag_rmse, 4), "real", round(float(real.mean()), 4),
          "baseline", round(float(base.mean()), 4), "nonfinite", log["nonfinite"],
          "grad max", f"{max(log['grad']):.2e}", flush=True)
