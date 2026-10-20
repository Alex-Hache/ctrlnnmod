"""
Géométrie induite par une paramétrisation, en descente de gradient pure.

Une descente sur θ déplace W = Φ(θ) de −η J Jᵀ ∇L (J = DΦ(θ)) : chaque paramétrisation impose sa
métrique. Ce script (1) mesure l'« empreinte » de J Jᵀ (rang, conditionnement κ) et (2) compare
les vitesses de convergence en descente de gradient pure.

Matrices définies positives (perte ½‖H − H*‖²_F, départ en H₀ = I) :
  euclid   : H libre, projection sur {H ⪰ εI} après chaque pas (descente projetée)
  bm       : H = XᵀX + εI (facteur de Burer–Monteiro, REN de ctrlnnmod)   poids 2(λᵢ + λⱼ)
  chol     : H = LLᵀ + εI, L triangulaire inférieure                        (non invariant)
  cholexp  : idem avec diag(L) = exp(d)
  exps     : H = exp(S), S symétrique                                        poids L(λᵢ, λⱼ)² (moyenne log)
  geo_expm, geo_cayley : geotorch.positive_definite (H = Q diag(softplus(l)) Qᵀ, Q par expm / Cayley)
                         poids hors diagonale 2(λᵢ − λⱼ)² : nuls si valeurs propres égales
Matrices orthogonales (perte ½‖Q − Q*‖²_F, départ en Q = I, Q* = rotations d'angle φ*) :
  geo_expm, geo_cayley : geotorch.orthogonal ; polar : Q libre + rétraction polaire
  Prédiction SO(2) : expm isométrique (dφ/dθ = 1) ; Cayley dφ/dθ = cos²(φ/2) → 0 quand φ → π.

Usage : python geometry.py psd --sweep   |   python geometry.py orth --sweep
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
from torch.func import jacrev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import geotorch_custom as geo  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.environ.get("OUT_DIR", "experiments/param_geometry/results")
EPS = 1e-6


def sym(A):
    return 0.5 * (A + A.T)


def sqrtm_psd(H):
    lam, Q = torch.linalg.eigh(H)
    return (Q * lam.clamp(min=0).sqrt()) @ Q.T


def logm_pd(H):
    lam, Q = torch.linalg.eigh(H)
    return (Q * lam.log()) @ Q.T


# ----------------------------------------------------------------------------------- PSD
class PSDParam(nn.Module):
    """θ ↦ H pour chaque paramétrisation ; initialisée pour que H(θ₀) = H₀."""

    def __init__(self, kind, H0, triv=None):
        super().__init__()
        self.kind, n = kind, H0.shape[0]
        self.n = n
        if kind == "euclid":
            self.t = nn.Parameter(H0.clone())
        elif kind == "bm":
            self.t = nn.Parameter(sqrtm_psd(H0 - EPS * torch.eye(n)))
        elif kind == "chol":
            self.t = nn.Parameter(torch.linalg.cholesky(H0 - EPS * torch.eye(n)))
        elif kind == "cholexp":
            L = torch.linalg.cholesky(H0 - EPS * torch.eye(n))
            self.t = nn.Parameter(torch.tril(L, -1) + torch.diag(L.diagonal().log()))
        elif kind == "exps":
            self.t = nn.Parameter(logm_pd(H0))
        elif kind.startswith("geo"):
            self.w = nn.Module()
            self.w.weight = nn.Parameter(torch.eye(n))
            geo.positive_definite(self.w, "weight", triv=kind.split("_")[1])
            self.w.weight = H0
        self.mask = torch.tril(torch.ones(n, n))

    def H(self):
        k, t, n = self.kind, getattr(self, "t", None), self.n
        if k == "euclid":
            return sym(t)
        if k == "bm":
            return t.T @ t + EPS * torch.eye(n)
        if k == "chol":
            L = t * self.mask
            return L @ L.T + EPS * torch.eye(n)
        if k == "cholexp":
            L = torch.tril(t, -1) + torch.diag(t.diagonal().exp())
            return L @ L.T + EPS * torch.eye(n)
        if k == "exps":
            return torch.linalg.matrix_exp(sym(t))
        return self.w.weight

    @torch.no_grad()
    def project(self):
        if self.kind == "euclid":
            lam, Q = torch.linalg.eigh(sym(self.t))
            self.t.copy_((Q * lam.clamp(min=EPS)) @ Q.T)


def fingerprint(model):
    """Spectre de J Jᵀ restreint aux matrices symétriques (coordonnées vech orthonormées)."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    names = list(params)
    n = model.n
    iu = torch.triu_indices(n, n)
    wgt = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0))      # vech isométrique

    def f(*vals):
        out = torch.func.functional_call(model, dict(zip(names, vals)), (), {}) if False else None
        return out
    flat = torch.cat([params[k].reshape(-1) for k in names])

    def h_of(v):
        i, d = 0, {}
        for k in names:
            m = params[k].numel()
            d[k] = v[i:i + m].view_as(params[k])
            i += m
        H = torch.func.functional_call(model, d, (), {}, strict=False, tie_weights=False) if False else None
        return H
    # functional_call sur la méthode H : on passe par un module enveloppe
    class Wrap(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s):
            return s.m.H()
    wrap = Wrap(model)

    def vechH(v):
        i, d = 0, {}
        for k in names:
            m = params[k].numel()
            d["m." + k] = v[i:i + m].view_as(params[k])
            i += m
        H = torch.func.functional_call(wrap, d, ())
        return H[iu[0], iu[1]] * wgt
    J = jacrev(vechH)(flat)
    ev = torch.linalg.eigvalsh(J @ J.T)
    tol = 1e-10 * ev[-1]
    return dict(rank=int((ev > tol).sum()), dim=int(ev.numel()), ev_min=float(ev[0]), ev_max=float(ev[-1]),
                kappa=float(ev[-1] / ev[0].clamp(min=1e-300)))


def psd_target(name, n, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = {"wellcond": torch.logspace(0, 1, n), "illcond": torch.logspace(-3, 0, n),
           "clustered": torch.tensor([1.0] * (n // 2) + [2.0] * (n - n // 2)),
           "nearbdry": torch.logspace(-4, -1, n)}[name]
    return (Q * lam) @ Q.T


def run_psd(kind, target, lr, seed, n=6, iters=20000, tol=1e-10):
    Hs = psd_target(target, n, seed)
    m = PSDParam(kind, torch.eye(n))
    fp0 = fingerprint(m)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    nrm = float((Hs ** 2).sum())
    log, it_tol, status, t0 = [], None, "ok", time.time()
    for it in range(iters):
        L = 0.5 * ((m.H() - Hs) ** 2).sum()
        rel = float(L.detach()) * 2 / nrm
        if not math.isfinite(rel):
            status = f"diverged@{it}"
            break
        if it % 20 == 0:
            log.append(rel)
        if rel < tol:
            it_tol = it
            break
        opt.zero_grad()
        L.backward()
        opt.step()
        m.project()
    final = rel
    fpS = None
    if status == "ok":
        try:
            fpS = fingerprint(m)
        except Exception as e:  # noqa: BLE001
            fpS = dict(error=repr(e)[:100])
    return dict(task="psd", kind=kind, target=target, lr=lr, seed=seed, status=status, it_tol=it_tol,
                final=final, log=log, fp_init=fp0, fp_final=fpS, time=time.time() - t0)


# ----------------------------------------------------------------------------------- SO(n)
def rot_target(phi, n=4):
    Q = torch.eye(n)
    for k in range(n // 2):
        c, s = math.cos(phi), math.sin(phi)
        Q[2 * k:2 * k + 2, 2 * k:2 * k + 2] = torch.tensor([[c, -s], [s, c]])
    return Q


class OrthParam(nn.Module):
    def __init__(self, kind, n=4):
        super().__init__()
        self.kind = kind
        if kind == "polar":
            self.t = nn.Parameter(torch.eye(n))
        else:
            self.w = nn.Module()
            self.w.weight = nn.Parameter(torch.eye(n))
            geo.orthogonal(self.w, "weight", triv=kind.split("_")[1])
            self.w.weight = torch.eye(n)

    def Q(self):
        return self.t if self.kind == "polar" else self.w.weight

    @torch.no_grad()
    def project(self):
        if self.kind == "polar":
            U, _, Vh = torch.linalg.svd(self.t)
            self.t.copy_(U @ Vh)


def run_orth(kind, phi_frac, lr, seed, n=4, iters=20000, tol=1e-10):
    torch.manual_seed(seed)
    Qs = rot_target(phi_frac * math.pi, n)
    m = OrthParam(kind, n)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    log, it_tol, status, t0 = [], None, "ok", time.time()
    for it in range(iters):
        L = 0.5 * ((m.Q() - Qs) ** 2).sum()
        rel = float(L.detach()) / n
        if not math.isfinite(rel):
            status = f"diverged@{it}"
            break
        if it % 20 == 0:
            log.append(rel)
        if rel < tol:
            it_tol = it
            break
        opt.zero_grad()
        L.backward()
        opt.step()
        m.project()
    return dict(task="orth", kind=kind, phi_frac=phi_frac, lr=lr, seed=seed, status=status, it_tol=it_tol,
                final=rel, log=log, time=time.time() - t0)


LRS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def _job(a):
    torch.set_num_threads(1)
    if a[0] == "psd":
        r = run_psd(*a[1:])
        name = f"psd_{a[1]}_{a[2]}_s{a[4]}_lr{a[3]}"
    else:
        r = run_orth(*a[1:])
        name = f"orth_{a[1]}_phi{a[2]}_s{a[4]}_lr{a[3]}"
    json.dump(r, open(f"{OUT}/{name}.json", "w"))
    return f"{name}: {r['status']} it_tol={r['it_tol']} final={r['final']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["psd", "orth"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.task == "psd":
        jobs = [("psd", k, t, lr, s) for k, t, lr, s in itertools.product(
            ("euclid", "bm", "chol", "cholexp", "exps", "geo_expm", "geo_cayley"),
            ("wellcond", "illcond", "clustered", "nearbdry"), LRS, (0, 1))]
    else:
        jobs = [("orth", k, p, lr, s) for k, p, lr, s in itertools.product(
            ("geo_expm", "geo_cayley", "polar"), (0.1, 0.5, 0.8, 0.95), LRS, (0, 1))]
    if a.part != "all":
        i, N = (int(v) for v in a.part.split("/"))
        jobs = jobs[i::N]
    from tqdm import tqdm
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc=f"géométrie {a.task}", mininterval=10):
            tqdm.write(line)
