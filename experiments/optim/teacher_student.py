"""
Pourquoi les modèles contraints apprennent-ils mal par descente de gradient ?
Étude maître–élève, REN explicite contractant (ᾱ = 1).

Maître : un REN dont la marge de certificat est réglée par ρ (rayon spectral de A). Quand
ρ → 1, le maître est près du bord de l'ensemble certifié. Il est DANS l'ensemble de l'élève :
le prior est vrai, donc un écart ne vient pas d'un manque d'expressivité.

Élèves (même fonction de départ) :
  ren_gd, ren_adam     : paramétrisation directe θ = (X, Y, …), descente sur θ (pratique usuelle)
  free_gd, free_adam   : mêmes équations, matrices W = (A, B_w, B_u, C1, …) libres, sans garantie
  riem_gd, riem_adam   : descente riemannienne sur l'ensemble certifié, métrique euclidienne en W,
                         rétraction donnée par la paramétrisation :
                             θ⁺ = θ + Jᵀ (J Jᵀ + λI)⁻¹ s_W,  J = DΦ(θ),  s_W = pas de free_*
                         Au 1er ordre, W bouge comme free_* ; chaque itéré reste certifié.

Prédiction (géométrie induite) : la descente sur θ déplace W de −η J Jᵀ ∇_W L. Pour H = XᵀX + εI,
elle induit Ḣ = −2(GM + MG), M = XᵀX : la vitesse s'effondre dans les directions où M est
petite, donc près du bord. L'écart ren_* / free_* doit croître quand ρ → 1, et riem_* le combler.

Usage : python teacher_student.py --rho 0.97 --seed 0 --method riem_gd --lr 0.1 --iters 1500
        python teacher_student.py --sweep --rhos 0.5,0.9,0.97,0.995 --seeds 0,1   (grille complète)
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time

import torch
from torch.func import jacrev

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN

torch.set_default_dtype(torch.float64)
OUT = os.environ.get("OUT_DIR", "experiments/optim/results")
NU, NY, NX, NQ = 2, 2, 8, 16
WKEYS = ("A", "Bw", "Bu", "C1", "D12", "lam", "C2", "D21", "D22", "bx", "bv", "by")


# ----------------------------------------------------------------------------------- modèle
def phi(p, nx, nq, eps=1e-3, ab=1.0):
    """Paramétrisation directe (ExplicitREN, contraction seule) : θ ↦ W. Fonction pure."""
    N = 2 * nx + nq
    H = p["X"].T @ p["X"] + eps * torch.eye(N, dtype=p["X"].dtype)
    H11, H21, H22 = H[:nx, :nx], H[nx:nx + nq, :nx], H[nx:nx + nq, nx:nx + nq]
    F, B1, P = H[nx + nq:, :nx], H[nx + nq:, nx:nx + nq], H[nx + nq:, nx + nq:]
    E = 0.5 * (H11 + P / ab ** 2 + p["Y"] - p["Y"].T)
    Ei = torch.linalg.inv(E)
    return dict(A=Ei @ F, Bw=Ei @ B1, Bu=Ei @ p["B2"], C1=-H21, D12=p["D12"], lam=0.5 * H22.abs().sum(1),
                C2=p["C2"], D21=p["D21"], D22=p["D22"], bx=Ei @ p["bx"], bv=p["bv"], by=p["by"])


def sim(W, u):
    """u : (B, T, nu) → y : (B, T, ny). x₀ = 0."""
    B, T, _ = u.shape
    x = torch.zeros(B, W["A"].shape[0], dtype=u.dtype)
    lam = W["lam"].clamp(min=1e-6)
    ys = []
    for t in range(T):
        w = torch.tanh((x @ W["C1"].T + u[:, t] @ W["D12"].T + W["bv"]) / lam)
        ys.append(x @ W["C2"].T + w @ W["D21"].T + u[:, t] @ W["D22"].T + W["by"])
        x = x @ W["A"].T + w @ W["Bw"].T + u[:, t] @ W["Bu"].T + W["bx"]
    return torch.stack(ys, 1)


def flat(d, keys):
    return torch.cat([d[k].reshape(-1) for k in keys])


def unflat(v, like, keys):
    out, i = {}, 0
    for k in keys:
        n = like[k].numel()
        out[k] = v[i:i + n].view_as(like[k])
        i += n
    return out


# ----------------------------------------------------------------------------------- maître
def teacher(rho, seed, nx=NX, nq=NQ, eps=1e-3):
    """Maître dans l'ensemble certifié de l'élève, de marge réglée par ρ.
    P = I, E = c I (c = 1/ρ², optimal pour la partie linéaire), F = c A, A = ρ Q (Q orthogonale),
    H22 = 2Λ diagonal. La partie non linéaire (C1, B_w, échelle s) est réduite jusqu'à ce qu'elle
    consomme au plus la moitié de la marge linéaire : la marge reste fixée par ρ."""
    g = torch.Generator().manual_seed(1000 + seed)
    Q, _ = torch.linalg.qr(torch.randn(nx, nx, generator=g))
    A = rho * Q
    C10 = torch.randn(nq, nx, generator=g) / math.sqrt(nx)
    Bw0 = torch.randn(nx, nq, generator=g) / math.sqrt(nq)
    lam = torch.ones(nq)
    c = 1 / rho ** 2

    def build(s):
        C1, B1, F = s * C10, c * s * Bw0, c * A
        H = torch.zeros(2 * nx + nq, 2 * nx + nq)
        H[:nx, :nx] = 2 * c * torch.eye(nx) - torch.eye(nx)
        H[nx:nx + nq, :nx], H[:nx, nx:nx + nq] = -C1, -C1.T
        H[nx:nx + nq, nx:nx + nq] = torch.diag(2 * lam)
        H[nx + nq:, :nx], H[:nx, nx + nq:] = F, F.T
        H[nx + nq:, nx:nx + nq], H[nx:nx + nq, nx + nq:] = B1, B1.T
        H[nx + nq:, nx + nq:] = torch.eye(nx)
        return H
    lin = float(torch.linalg.eigvalsh(build(0.0))[0])
    for k in range(80):
        s = 0.85 ** k
        H = build(s)
        ev = torch.linalg.eigvalsh(H)
        if ev[0] >= 0.5 * lin:
            break
    X = torch.linalg.cholesky(H - eps * torch.eye(H.shape[0])).T           # XᵀX = H − εI
    p = dict(X=X, Y=torch.zeros(nx, nx), B2=c * torch.randn(nx, NU, generator=g) / math.sqrt(NU),
             D12=torch.randn(nq, NU, generator=g) / math.sqrt(NU), C2=torch.randn(NY, nx, generator=g) / math.sqrt(nx),
             D21=0.3 * torch.randn(NY, nq, generator=g) / math.sqrt(nq), D22=torch.zeros(NY, NU),
             bx=torch.zeros(nx), bv=torch.zeros(nq), by=torch.zeros(NY))
    info = dict(rho=rho, nonlin_scale=float(s), margin=float(ev[0] / ev[-1]), lam_min_H=float(ev[0]),
                lam_min_H_linear=lin)
    return p, info


def inputs(n, T, seed):
    """Entrées lisses : bruit blanc filtré (premier ordre, constante aléatoire)."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(n, T, NU, generator=g)
    a = torch.rand(n, 1, NU, generator=g) * 0.9
    u = torch.zeros_like(e)
    for t in range(T):
        u[:, t] = (a[:, 0] * u[:, t - 1] if t else 0) + (1 - a[:, 0] ** 2).sqrt() * e[:, t]
    return u


# ----------------------------------------------------------------------------------- élèves
def student_init(seed, nx=NX, nq=NQ):
    torch.manual_seed(seed)
    m = ExplicitREN(NU, NY, nx, nq, gamma=None, alpha_bar=1.0).double()
    return {k: v.detach().clone() for k, v in m.named_parameters()}


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, b1, b2, eps, 0, 0, 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return -self.lr * mh / (vh.sqrt() + self.eps)


def run(rho, seed, method, lr, iters=1500, log_every=25, damping=1e-6, nx=NX, nq=NQ):
    tp, tinfo = teacher(rho, seed)
    Wt = phi(tp, NX, NQ)
    u_tr, u_te = inputs(256, 100, 10 + seed), inputs(64, 400, 20 + seed)
    with torch.no_grad():
        y_tr, y_te = sim(Wt, u_tr), sim(Wt, u_te)
    sc = y_tr.std()
    y_tr, y_te = y_tr / sc, y_te / sc
    Wt = {**Wt, "C2": Wt["C2"] / sc, "D21": Wt["D21"] / sc, "D22": Wt["D22"] / sc, "by": Wt["by"] / sc}

    th0 = student_init(seed, nx, nq)
    TK = list(th0.keys())
    kind, opt_name = method.split("_")
    if kind == "free":
        z = {k: v.clone() for k, v in phi(th0, nx, nq).items()}
        keys, x = WKEYS, flat(z, WKEYS)
        like = z
    else:
        keys, x, like = TK, flat(th0, TK), th0
    opt = Adam(lr) if opt_name == "adam" else None
    phiflat = lambda v: flat(phi(unflat(v, th0, TK), nx, nq), WKEYS)          # noqa: E731
    Wlike = phi(th0, nx, nq)
    loss_W = lambda Wv, u, y: ((sim(unflat(Wv, Wlike, WKEYS), u) - y) ** 2).mean()  # noqa: E731

    log = dict(it=[], train=[], test=[], cond_M=[], lmin_M=[], cond_JJt=[], margin=[], retr_err=[])
    t0, status = time.time(), "ok"
    for it in range(iters + 1):
        if kind == "free":
            xv = x.detach().requires_grad_(True)
            L = loss_W(xv, u_tr, y_tr)
            g, = torch.autograd.grad(L, xv)
            step = -lr * g if opt is None else opt.step(g)
            x_new = x + step
        elif kind == "ren":
            xv = x.detach().requires_grad_(True)
            L = loss_W(phiflat(xv), u_tr, y_tr)
            g, = torch.autograd.grad(L, xv)
            step = -lr * g if opt is None else opt.step(g)
            x_new = x + step
        else:                                                                   # riem
            Wv = phiflat(x).detach().requires_grad_(True)
            L = loss_W(Wv, u_tr, y_tr)
            gW, = torch.autograd.grad(L, Wv)
            sW = -lr * gW if opt is None else opt.step(gW)
            J = jacrev(phiflat)(x)
            JJ = J @ J.T
            lamd = damping * torch.linalg.eigvalsh(JJ)[-1]
            v = torch.linalg.solve(JJ + lamd * torch.eye(JJ.shape[0]), sW)
            x_new = x + J.T @ v
        if not torch.isfinite(L) or not torch.isfinite(x_new).all():
            status = f"diverged@{it}"
            break
        if it % log_every == 0:
            with torch.no_grad():
                W = unflat(x, like, keys) if kind == "free" else phi(unflat(x, th0, TK), nx, nq)
                te = float(((sim(W, u_te) - y_te) ** 2).mean())
            log["it"].append(it)
            log["train"].append(float(L.detach()))
            log["test"].append(te)
            if kind != "free":
                p = unflat(x, th0, TK)
                evM = torch.linalg.eigvalsh(p["X"].T @ p["X"])
                log["cond_M"].append(float(evM[-1] / evM[0].clamp(min=1e-300)))
                log["lmin_M"].append(float(evM[0]))
                N = 2 * nx + nq
                evH = torch.linalg.eigvalsh(p["X"].T @ p["X"] + 1e-3 * torch.eye(N))
                log["margin"].append(float(evH[0] / evH[-1]))
                if it % (4 * log_every) == 0 or kind == "riem":
                    Jd = J if kind == "riem" else jacrev(phiflat)(x)
                    ev = torch.linalg.eigvalsh(Jd @ Jd.T)
                    log["cond_JJt"].append(float(ev[-1] / ev[0].clamp(min=1e-300)))
                if kind == "riem":
                    with torch.no_grad():
                        dW = phiflat(x_new) - Wv.detach()
                        log["retr_err"].append(float((dW - sW).norm() / sW.norm().clamp(min=1e-300)))
        x = x_new.detach()
    return dict(rho=rho, seed=seed, method=method, lr=lr, iters=iters, status=status, teacher=tinfo,
                time=time.time() - t0, log=log, final_train=log["train"][-1] if log["train"] else None,
                final_test=log["test"][-1] if log["test"] else None)


GRID = {"ren_gd": [0.01, 0.03, 0.1, 0.3, 1.0], "free_gd": [0.01, 0.03, 0.1, 0.3, 1.0],
        "riem_gd": [0.01, 0.03, 0.1, 0.3, 1.0], "ren_adam": [1e-3, 3e-3, 1e-2, 3e-2],
        "free_adam": [1e-3, 3e-3, 1e-2, 3e-2], "riem_adam": [1e-3, 3e-3, 1e-2, 3e-2]}


def _job(args):
    torch.set_num_threads(1)
    rho, seed, method, lr, iters = args
    r = run(rho, seed, method, lr, iters)
    name = f"ts_{method}_rho{rho}_s{seed}_lr{lr}"
    json.dump(r, open(f"{OUT}/{name}.json", "w"))
    return f"{name}: {r['status']} train={r['final_train']:.2e} test={r['final_test']:.2e} {r['time']:.0f}s" \
        if r["final_train"] is not None else f"{name}: {r['status']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="riem_gd")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rhos", default="0.5,0.9,0.97,0.995")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--methods", default=",".join(GRID))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.sweep:
        print(_job((a.rho, a.seed, a.method, a.lr, a.iters)))
    else:
        from tqdm import tqdm
        jobs = [(r, s, m, lr, a.iters) for r, s, m in itertools.product(
            [float(v) for v in a.rhos.split(",")], [int(v) for v in a.seeds.split(",")], a.methods.split(","))
            for lr in GRID[m]]
        jobs.sort(key=lambda j: j[2].startswith("riem"), reverse=True)        # les plus longs d'abord
        nproc = int(os.environ.get("NPROC", os.cpu_count() or 1))
        with mp.get_context("fork").Pool(nproc) as pool:
            for line in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc="maître–élève", mininterval=10):
                tqdm.write(line)
