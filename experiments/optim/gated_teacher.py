"""
Expressivité ou optimisation ? Maître à portes (type GRU / Mamba), contractant.

Le maître : z = σ(W_z x + U_z u), h̃ = tanh(W_h x + U_h u), x⁺ = (1 − z) ⊙ x + z ⊙ h̃, y = C x,
avec ‖W_h‖₂ = 0,9 et une porte surtout pilotée par l'entrée (sélective). Il contracte, mais il
n'est PAS dans la classe REN explicite (pas de produit état × porte).

Élèves (Adam, même budget) :
  ren   : REN explicite certifié (θ = X, Y, …)          — classe sans portes, garantie
  free  : mêmes équations que ren, matrices libres         — classe sans portes, sans garantie
  sren  : REN sélectif certifié (portes par tête)          — classe à portes, garantie
  gru   : GRUCell libre                                    — classe à portes, sans garantie

Lecture : gru ≪ free ≈ ren ⇒ l'écart vient de la classe (expressivité), pas de la contrainte.
          free ≪ ren                ⇒ l'écart vient de l'optimisation sous contrainte.
Usage : python gated_teacher.py --sweep --seeds 0,1,2 --widths 1,4
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time

import torch
from torch import nn

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
from ctrlnmod.models.discrete.selective_ren import SelectiveREN

torch.set_default_dtype(torch.float64)
OUT = os.environ.get("OUT_DIR", "experiments/optim/results_gated")
NU, NY, NX = 2, 2, 8


def teacher_params(seed, gate_in=3.0):
    g = torch.Generator().manual_seed(2000 + seed)
    Wh = torch.randn(NX, NX, generator=g)
    Wh = 0.9 * Wh / torch.linalg.matrix_norm(Wh, 2)
    return dict(Wz=0.3 * torch.randn(NX, NX, generator=g) / math.sqrt(NX),
                Uz=gate_in * torch.randn(NX, NU, generator=g) / math.sqrt(NU),
                bz=torch.full((NX,), -1.0), Wh=Wh, Uh=torch.randn(NX, NU, generator=g) / math.sqrt(NU),
                C=torch.randn(NY, NX, generator=g) / math.sqrt(NX))


def teacher_sim(p, u):
    B, T, _ = u.shape
    x, ys = torch.zeros(B, NX), []
    for t in range(T):
        ys.append(x @ p["C"].T)
        z = torch.sigmoid(x @ p["Wz"].T + u[:, t] @ p["Uz"].T + p["bz"])
        h = torch.tanh(x @ p["Wh"].T + u[:, t] @ p["Uh"].T)
        x = (1 - z) * x + z * h
    return torch.stack(ys, 1)


def inputs(n, T, seed):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(n, T, NU, generator=g)
    a = torch.rand(n, 1, NU, generator=g) * 0.9
    u = torch.zeros_like(e)
    for t in range(T):
        u[:, t] = (a[:, 0] * u[:, t - 1] if t else 0) + (1 - a[:, 0] ** 2).sqrt() * e[:, t]
    return u


class FreeREN(nn.Module):
    """Mêmes équations que ExplicitREN (D11 = 0), matrices libres, initialisées depuis un REN."""

    def __init__(self, ren):
        super().__init__()
        fr = {k: v.detach().clone() for k, v in ren.frame().items()}
        for k in ("A", "Bw", "Bu", "C1", "bx"):
            setattr(self, k, nn.Parameter(fr[k]))
        self.log_lam = nn.Parameter(fr["Lambda"].clamp(min=1e-3).log())
        for k in ("D12", "C2", "D21", "D22", "bv", "by"):
            setattr(self, k, nn.Parameter(getattr(ren, k).detach().clone()))

    def forward(self, u):
        B, T, _ = u.shape
        x, ys, lam = torch.zeros(B, self.A.shape[0]), [], self.log_lam.exp()
        for t in range(T):
            w = torch.tanh((x @ self.C1.T + u[:, t] @ self.D12.T + self.bv) / lam)
            ys.append(x @ self.C2.T + w @ self.D21.T + u[:, t] @ self.D22.T + self.by)
            x = x @ self.A.T + w @ self.Bw.T + u[:, t] @ self.Bu.T + self.bx
        return torch.stack(ys, 1)


class RENOut(nn.Module):
    def __init__(self, nx):
        super().__init__()
        self.m = ExplicitREN(NU, NY, nx, 2 * nx, gamma=None, alpha_bar=1.0)

    def forward(self, u):
        y, _ = self.m(u)
        return y


class SRENOut(nn.Module):
    def __init__(self, nx):
        super().__init__()
        heads = max(1, nx // 4)
        self.m = SelectiveREN(NU, n_heads=heads, nx=4, nq=8, alpha_bar=1.0, g_min=0.02)
        self.C = nn.Linear(heads * 4, NY)

    def forward(self, u):
        from ctrlnmod.models.discrete.selective_ren import rollout
        xs = rollout(self.m, u, None, "torch")                     # états après chaque pas
        x0 = torch.zeros(u.shape[0], 1, xs.shape[-1])
        return self.C(torch.cat([x0, xs[:, :-1]], 1))               # y_t = C x_t, comme le maître


class GRUOut(nn.Module):
    def __init__(self, nx):
        super().__init__()
        self.cell, self.C = nn.GRUCell(NU, nx), nn.Linear(nx, NY)

    def forward(self, u):
        B, T, _ = u.shape
        h, ys = torch.zeros(B, self.cell.hidden_size), []
        for t in range(T):
            ys.append(self.C(h))
            h = self.cell(u[:, t], h)
        return torch.stack(ys, 1)


def make(kind, nx, seed):
    torch.manual_seed(seed)
    if kind == "ren":
        return RENOut(nx)
    if kind == "free":
        return FreeREN(RENOut(nx).m)
    if kind == "sren":
        return SRENOut(nx)
    return GRUOut(nx)


def run(seed, kind, lr, width, iters=1500, log_every=25):
    p = teacher_params(seed)
    u_tr, u_te = inputs(256, 100, 30 + seed), inputs(64, 400, 40 + seed)
    with torch.no_grad():
        y_tr, y_te = teacher_sim(p, u_tr), teacher_sim(p, u_te)
    sc = y_tr.std()
    y_tr, y_te = y_tr / sc, y_te / sc
    m = make(kind, NX * width, seed).double()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    log, t0, status = dict(it=[], train=[], test=[]), time.time(), "ok"
    for it in range(iters + 1):
        L = ((m(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(L):
            status = f"diverged@{it}"
            break
        if it % log_every == 0:
            with torch.no_grad():
                log["it"].append(it)
                log["train"].append(float(L))
                log["test"].append(float(((m(u_te) - y_te) ** 2).mean()))
        opt.zero_grad()
        L.backward()
        opt.step()
    return dict(seed=seed, kind=kind, lr=lr, width=width, status=status, time=time.time() - t0,
                nparams=sum(q.numel() for q in m.parameters()), log=log,
                final_train=log["train"][-1] if log["train"] else None,
                final_test=log["test"][-1] if log["test"] else None)


def _job(args):
    torch.set_num_threads(1)
    seed, kind, lr, width, iters = args
    r = run(seed, kind, lr, width, iters)
    name = f"gt_{kind}_w{width}_s{seed}_lr{lr}"
    json.dump(r, open(f"{OUT}/{name}.json", "w"))
    return f"{name}: {r['status']} train={r['final_train']} test={r['final_test']} {r['time']:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--widths", default="1,4")
    ap.add_argument("--kinds", default="ren,free,sren,gru")
    ap.add_argument("--iters", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, k, lr, w, a.iters) for s, k, w in itertools.product(
        [int(v) for v in a.seeds.split(",")], a.kinds.split(","), [int(v) for v in a.widths.split(",")])
        for lr in (1e-3, 3e-3, 1e-2, 3e-2)]
    from tqdm import tqdm
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc="maître à portes", mininterval=10):
            tqdm.write(line)
