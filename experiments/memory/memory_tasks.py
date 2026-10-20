"""
Mémoire certifiée : que peut se rappeler un modèle certifié, et quand ?

M1 — rappel retardé (théorème A). u_0 = v ∈ {−1, +1}, puis u_t = 0 ; cible : y_k = v.
  Un REN de gain incrémental γ et de taux ᾱ vérifie |Δy_k| ≤ γ ᾱᵏ |Δu_0| (Lean :
  `These.pulse_forgetting`). Donc l'erreur quadratique est au moins (1 − γᾱᵏ)²₊, QUELLE QUE SOIT
  la largeur. Seuil prédit : k* = ln γ / ln(1/ᾱ).

M2 — rappel sélectif avec distracteurs (théorèmes B et C). Canal 0 : une valeur N(0, 1) à chaque
  pas (distracteurs) ; canal 1 : un marqueur à un instant t_m ∈ [0, K/2) ; un pas de requête à la
  fin. Cible : la valeur marquée. Il faut une porte pilotée par l'entrée (garder quand le
  marqueur est là, ignorer sinon).

Modèles :
  ren      : REN explicite certifié (γ, ᾱ)                 — certificat uniforme
  free     : mêmes équations, matrices libres                — aucun certificat
  sren     : REN sélectif (une porte par tête de 8 états)    — contraction pour toute suite de portes
  linf     : RNN à portes par unité, certifié en ℓ∞ (nouveau) — contraction pour toute suite de portes
  gru, lru, mamba2                                          — références
Usage : python memory_tasks.py m1 --sweep   |   python memory_tasks.py m2 --sweep
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "world_model"))
from ctrlnmod.models.discrete.explicit_ren import ExplicitREN  # noqa: E402
from ctrlnmod.models.discrete.selective_ren import SelectiveREN, rollout  # noqa: E402

OUT = os.environ.get("OUT_DIR", "experiments/memory/results")


# ------------------------------------------------------------------------------------ modèles
class RENm(nn.Module):
    def __init__(self, nu, n, gamma=None, ab=1.0):
        super().__init__()
        self.m = ExplicitREN(nu, 1, n, 2 * n, gamma=gamma, alpha_bar=ab)

    def forward(self, u):
        return self.m(u)[0]


class FreeREN(nn.Module):
    """Mêmes équations que ExplicitREN (D11 = 0), sans certificat, partant du même modèle."""

    def __init__(self, nu, n):
        super().__init__()
        ren = ExplicitREN(nu, 1, n, 2 * n, gamma=None, alpha_bar=1.0)
        fr = {k: v.detach().clone() for k, v in ren.frame().items()}
        for k in ("A", "Bw", "Bu", "C1", "bx"):
            setattr(self, k, nn.Parameter(fr[k]))
        self.log_lam = nn.Parameter(fr["Lambda"].clamp(min=1e-3).log())
        for k in ("D12", "C2", "D21", "D22", "bv", "by"):
            setattr(self, k, nn.Parameter(getattr(ren, k).detach().clone()))

    def forward(self, u):
        B, T, _ = u.shape
        x, ys, lam = u.new_zeros(B, self.A.shape[0]), [], self.log_lam.exp()
        for t in range(T):
            w = torch.tanh((x @ self.C1.T + u[:, t] @ self.D12.T + self.bv) / lam)
            ys.append(x @ self.C2.T + w @ self.D21.T + u[:, t] @ self.D22.T + self.by)
            x = x @ self.A.T + w @ self.Bw.T + u[:, t] @ self.Bu.T + self.bx
        return torch.stack(ys, 1)


class SRENm(nn.Module):
    def __init__(self, nu, n, g_min=0.005):
        super().__init__()
        self.m = SelectiveREN(nu, n_heads=max(1, n // 8), nx=8, nq=16, alpha_bar=1.0, g_min=g_min)
        self.C = nn.Linear(self.m.n_state, 1)

    def forward(self, u):
        return self.C(rollout(self.m, u, None, "torch"))          # y_t lit l'état après u_t


class LinfGated(nn.Module):
    """RNN à portes PAR UNITÉ, certifié contractant en norme ℓ∞ pour toute suite de portes.

    x⁺ = (1 − z) ⊙ x + z ⊙ f(x, u),   z = z_min + (1 − z_min) σ(W_z u + b_z)   (porte d'entrée)
    f(x, u) = A x + B tanh(C x + D u + b) + E u + b_x
    Certificat : Σⱼ|Aᵢⱼ| + Σₖ|Bᵢₖ| Σⱼ|Cₖⱼ| ≤ ρ < 1 pour chaque ligne i (dominance diagonale
    généralisée). Alors |Δx⁺ᵢ| ≤ (1 − zᵢ(1 − ρ)) ‖Δx‖∞ (Lean : `These.gated_linf_step`).
    Paramétrisation directe : les lignes de [Ã | B̃] sont réduites d'un facteur ρ/(ρ + rᵢ)."""

    def __init__(self, nu, n, rho=0.99, z_min=1e-3):
        super().__init__()
        q = 2 * n
        self.rho, self.z_min = rho, z_min
        self.A_, self.B_ = nn.Parameter(torch.randn(n, n) / n), nn.Parameter(torch.randn(n, q) / q)
        self.C_ = nn.Parameter(torch.randn(q, n) / math.sqrt(n))
        self.D, self.E = nn.Parameter(torch.randn(q, nu)), nn.Parameter(torch.randn(n, nu) / math.sqrt(nu))
        self.b, self.bx = nn.Parameter(torch.zeros(q)), nn.Parameter(torch.zeros(n))
        self.gate = nn.Linear(nu, n)
        nn.init.constant_(self.gate.bias, -3.0)                    # portes plutôt fermées au départ
        self.C = nn.Linear(n, 1)

    def weights(self):
        c = self.C_.abs().sum(1)                                   # Σⱼ |Cₖⱼ|
        r = self.A_.abs().sum(1) + self.B_.abs() @ c                # somme de ligne avant réduction
        s = (self.rho / (self.rho + r))[:, None]
        return self.A_ * s, self.B_ * s, self.C_

    def forward(self, u):
        A, B, C = self.weights()
        Bt, T, _ = u.shape
        x, ys = u.new_zeros(Bt, A.shape[0]), []
        z = self.z_min + (1 - self.z_min) * torch.sigmoid(self.gate(u))
        for t in range(T):
            f = x @ A.T + torch.tanh(x @ C.T + u[:, t] @ self.D.T + self.b) @ B.T + u[:, t] @ self.E.T + self.bx
            x = (1 - z[:, t]) * x + z[:, t] * f
            ys.append(self.C(x))
        return torch.stack(ys, 1)

    @torch.no_grad()
    def row_sum(self):
        A, B, C = self.weights()
        return float((A.abs().sum(1) + B.abs() @ C.abs().sum(1)).max())


class BlockOut(nn.Module):
    """BlockGatedRNN (têtes de 2, une porte par tête, norme mixte max ℓ²) + lecture linéaire."""

    def __init__(self, nu, n, k=2, ny=1):
        super().__init__()
        from ctrlnmod.models.discrete.linf_gated import BlockGatedRNN
        self.m = BlockGatedRNN(nu, n // k, k=k)
        self.C = nn.Linear(n, ny)

    def forward(self, u):
        return self.C(self.m(u))


class GRUm(nn.Module):
    def __init__(self, nu, n):
        super().__init__()
        self.g, self.C = nn.GRU(nu, n, batch_first=True), nn.Linear(n, 1)

    def forward(self, u):
        return self.C(self.g(u)[0])


class LRUm(nn.Module):
    """Deux couches LRU (Orvieto et al., 2023) : récurrence linéaire diagonale stable, sans porte."""

    def __init__(self, nu, n, layers=2):
        super().__init__()
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pixel_wm"))
        from baselines import LRULayer
        self.inp, self.C = nn.Linear(nu, n), nn.Linear(n, 1)
        self.layers = nn.ModuleList([LRULayer(n, n, r_min=0.9, r_max=0.999) for _ in range(layers)])

    def forward(self, u):
        B, T, _ = u.shape
        sts = [(u.new_zeros(B, l.B_re.shape[0]),) * 2 for l in self.layers]
        ys = []
        for t in range(T):
            tok, new = self.inp(u[:, t]), []
            for lay, st in zip(self.layers, sts):
                st, tok = lay.step(st, tok)
                new.append(st)
            sts = new
            ys.append(self.C(tok))
        return torch.stack(ys, 1)


class Mamba2m(nn.Module):
    def __init__(self, nu, n, layers=2):
        super().__init__()
        from wm import Mamba2Block
        self.inp, self.C = nn.Linear(nu, n), nn.Linear(n, 1)
        self.blocks = nn.ModuleList([Mamba2Block(n) for _ in range(layers)])

    def forward(self, u):
        B, T, _ = u.shape
        sts = [b.init_state(B) for b in self.blocks]
        ys = []
        for t in range(T):
            tok, new = self.inp(u[:, t]), []
            for b, st in zip(self.blocks, sts):
                st, tok = b.step(st, tok)
                new.append(st)
            sts = new
            ys.append(self.C(tok))
        return torch.stack(ys, 1)


def make(kind, nu, n, gamma=None, ab=1.0):
    return {"ren": lambda: RENm(nu, n, gamma, ab), "free": lambda: FreeREN(nu, n), "sren": lambda: SRENm(nu, n),
            "linf": lambda: LinfGated(nu, n), "block": lambda: BlockOut(nu, n), "gru": lambda: GRUm(nu, n),
            "lru": lambda: LRUm(nu, n),
            "mamba2": lambda: Mamba2m(nu, n)}[kind]()


# ------------------------------------------------------------------------------------ tâches
def batch_m1(B, k, g):
    v = torch.randint(0, 2, (B,), generator=g).float() * 2 - 1
    u = torch.zeros(B, k + 1, 1)
    u[:, 0, 0] = v
    return u, v                                     # cible : y_k (k pas après l'impulsion)


def batch_m2(B, K, g):
    u = torch.zeros(B, K + 1, 2)
    u[:, :K, 0] = torch.randn(B, K, generator=g)    # distracteurs
    tm = torch.randint(0, K // 2, (B,), generator=g)
    u[torch.arange(B), tm, 1] = 1.0                 # marqueur
    return u, u[torch.arange(B), tm, 0]             # cible : la valeur marquée, lue au pas de requête K


def run(task, kind, n, lag, lr, seed, iters, gamma=None, ab=1.0, batch=256):
    torch.manual_seed(seed)
    nu = 1 if task == "m1" else 2
    if task == "m2":
        batch = 128
    m = make(kind, nu, n, gamma, ab)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(100 + seed)
    mk = batch_m1 if task == "m1" else batch_m2
    log, t0, status = dict(it=[], loss=[]), time.time(), "ok"
    for it in range(iters):
        u, y = mk(batch, lag, g)
        loss = ((m(u)[:, -1, 0] - y) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if it % 50 == 0:
            log["it"].append(it)
            log["loss"].append(float(loss.detach()))
    with torch.no_grad():
        u, y = mk(2048, lag, torch.Generator().manual_seed(999))
        pred = m(u)[:, -1, 0]
        test = float(((pred - y) ** 2).mean())
        extra = {}
        if task == "m1":
            p, mneg = pred[y > 0].mean(), pred[y < 0].mean()
            extra["gain_eff"] = float((p - mneg) / 2)     # |Δy_k| / |Δu_0| mesuré
        if kind == "linf":
            extra["row_sum"] = m.row_sum()
    floor = None
    if task == "m1" and gamma is not None:
        floor = max(0.0, 1 - gamma * ab ** lag) ** 2
    return dict(task=task, kind=kind, n=n, lag=lag, lr=lr, seed=seed, gamma=gamma, ab=ab, status=status,
                test=test, floor=floor, time=time.time() - t0, log=log,
                nparams=sum(q.numel() for q in m.parameters()), **extra)


def _job(a):
    torch.set_num_threads(1)
    r = run(*a)
    task, kind, n, lag, lr, seed, iters, gamma, ab = a
    name = f"{task}_{kind}_n{n}_k{lag}_g{gamma}_a{ab}_s{seed}_lr{lr}"
    json.dump(r, open(f"{OUT}/{name}.json", "w"))
    fl = f" plancher={r['floor']:.3f}" if r["floor"] is not None else ""
    return f"{name}: {r['status']} test={r['test']:.3f}{fl} {r['time']:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["m1", "m2"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--kinds", default=None)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    lrs = (1e-3, 3e-3, 1e-2)
    if a.task == "m1":
        jobs = [("m1", "ren", n, k, lr, s, a.iters, 4.0, ab)
                for ab, n, k, lr, s in itertools.product((0.8, 0.9, 0.95), (8, 32, 128), (2, 4, 8, 16, 32, 64), lrs, seeds)]
        jobs += [("m1", kd, 32, k, lr, s, a.iters, None, 1.0)
                 for kd, k, lr, s in itertools.product(("free", "gru"), (2, 4, 8, 16, 32, 64), lrs, seeds)]
    else:
        kinds = (a.kinds or "ren,free,sren,linf,gru,lru,mamba2").split(",")
        jobs = [("m2", kd, 32, K, lr, s, a.iters, None, 1.0)
                for kd, K, lr, s in itertools.product(kinds, (16, 64), lrs, seeds)]
    if a.part != "all":                                    # découpage entre conteneurs : "i/N"
        i, N = (int(v) for v in a.part.split("/"))
        jobs = jobs[i::N]
    from tqdm import tqdm
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc=f"mémoire {a.task}", mininterval=10):
            tqdm.write(line)
