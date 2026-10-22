"""
Q0 — M2-t : « la contrainte ferme la porte » ? Critères : `analysis/m2t.md` (6dafd7e).
  u1    : plateau ou lenteur au point final de P (T4) ;
  u2    : barrière sur le chemin Φ((1−λ)θ_P' + λθ_maître), têtes alignées ;
  train : U3 (pénalité λ·B̃·C, bras pen0.03, pen0.3) et U4 (cont, reopen, depuis P final).
Type par défaut float32 (le tirage du banc en dépend).
"""
import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402
import m2bis as mb  # noqa: E402
import m2q  # noqa: E402
import m2s  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2t"))
D4 = os.path.join(HERE, "results_m2q")
RHO, TOL = m2.RHO, m2.TOL
ARMS = ("pen0.03", "pen0.3", "cont", "reopen")


def load_p(seed):
    model = m2q.build4("p", seed)
    model.load_state_dict(torch.load(os.path.join(D4, f"t4_p_s{seed}.pt")))
    return model


def as_free(model):
    """Copie libre qui a exactement les poids effectifs W = Φ(θ) du modèle."""
    new = copy.deepcopy(model)
    with torch.no_grad():
        W = model.m.weights()
        new.m.__class__ = m2.FreeBlockGatedRNN
        new.m.A_.copy_(W["A"])
        new.m.B_.copy_(W["B"])
    return new


# ------------------------------------------------------------------ U1
def tangent_ratio(free, u, y, ref_norm):
    m = free.m
    loss = ((free(u) - y) ** 2).mean()
    X = [m.A_, m.B_, m.C]
    g = torch.autograd.grad(loss, X)
    gv = torch.cat([x.reshape(-1) for x in g])
    r = mb.r_raw(m.H, m.k, m.nq, *X)
    sat = [h for h in range(m.H) if float(r[h]) >= RHO - 1e-4]
    N = []
    for h in sat:
        gr = torch.autograd.grad(r[h], X, retain_graph=True, allow_unused=True)
        N.append(torch.cat([(a if a is not None else torch.zeros_like(x)).reshape(-1) for a, x in zip(gr, X)]))
    if N:
        Nm = torch.stack(N, 1).double()
        coef = torch.linalg.lstsq(Nm, gv.double()[:, None]).solution
        gt = gv.double() - (Nm @ coef)[:, 0]
    else:
        gt = gv.double()
    return float(gt.norm() / ref_norm), len(sat), float(gv.norm() / ref_norm)


def u1(seed):
    torch.set_num_threads(2)
    u, y, _, _ = m2.data(seed)
    init = mb.build("pint", seed)                                  # W₀' : référence ‖g(0)‖
    loss0 = ((init(u) - y) ** 2).mean()
    g0 = torch.autograd.grad(loss0, [init.m.A_, init.m.B_, init.m.C])
    ref = float(torch.cat([x.reshape(-1) for x in g0]).norm())
    free = as_free(load_p(seed))
    rg, nsat, rfull = tangent_ratio(free, u, y, ref)
    L = json.load(open(os.path.join(D4, f"t4_p_s{seed}.json")))["losses"]
    return dict(seed=seed, rho_g=rg, rho_full=rfull, n_sat=nsat, pente=(L[2500] - L[3000]) / L[2500],
                L2500=L[2500], L3000=L[3000])


# ------------------------------------------------------------------ U2
@torch.no_grad()
def states(model, u):
    return model.m.forward(u)


@torch.no_grad()
def align(free, teach, u):
    """Permutation des têtes + rotation orthogonale 2×2 par tête (Procrustes) de P vers la représentation du maître."""
    H, k = free.m.H, free.m.k
    Y = states(free, u).reshape(-1, H, k)
    X = states(teach, u).reshape(-1, H, k)
    free_heads, perm, Q = list(range(H)), [None] * H, [torch.eye(k)] * H
    for i in range(4):                                             # 4 oscillateurs utiles
        best = None
        for h in free_heads:
            sol = torch.linalg.lstsq(Y[:, h], X[:, i]).solution
            res = X[:, i] - Y[:, h] @ sol
            r2 = 1 - float((res ** 2).sum() / ((X[:, i] - X[:, i].mean(0)) ** 2).sum())
            if best is None or r2 > best[0]:
                best = (r2, h)
        _, h = best
        free_heads.remove(h)
        perm[i] = h
        U_, _, Vt = torch.linalg.svd(Y[:, h].T @ X[:, i])
        Q[i] = U_ @ Vt                                             # y_h Q ≈ x_i
    for i in range(4, H):
        perm[i] = free_heads[i - 4]
    Tf = torch.zeros(H * k, H * k)
    for i, h in enumerate(perm):
        Tf[i * k:(i + 1) * k, h * k:(h + 1) * k] = Q[i].T              # x'_i = Q_iᵀ x_h
    new = copy.deepcopy(free)
    m, Ti = new.m, torch.linalg.inv(Tf)
    m.A_.copy_(Tf @ m.A_ @ Ti)
    m.B_.copy_(Tf @ m.B_)
    m.C.copy_(m.C @ Ti)
    m.E.copy_(Tf @ m.E)
    m.bx.copy_(Tf @ m.bx)
    new.C.weight.copy_(new.C.weight @ Ti)
    m.gate_proj.weight.copy_(free.m.gate_proj.weight[perm])
    m.gate_proj.bias.copy_(free.m.gate_proj.bias[perm])
    return new, perm


@torch.no_grad()
def u2(seed):
    torch.set_num_threads(2)
    u, y, _, _ = m2.data(seed)
    free = as_free(load_p(seed))
    teach = mb.build("pint", seed)
    teach.m.__class__ = m2.FreeBlockGatedRNN
    m2s.teacher_on_boundary(teach, seed)
    al, perm = align(free, teach, u)
    io = float((al(u) - free(u)).abs().max() / free(u).abs().max())
    r0 = float(m2q.r_W(free.m).max())
    r1 = float(m2q.r_W(al.m).max())
    path = copy.deepcopy(al)
    path.m.__class__ = m2q.ExactProjBlock
    Pa = dict(al.named_parameters())
    Pt = dict(teach.named_parameters())
    Ls = []
    for j in range(21):
        lam = j / 20
        for n, p in path.named_parameters():
            p.copy_((1 - lam) * Pa[n] + lam * Pt[n])
        Ls.append(float(((path(u) - y) ** 2).mean()))
    return dict(seed=seed, perm=perm, io_err=io, r_before=r0, r_after=r1, L=Ls)


# ------------------------------------------------------------------ U3 / U4
def rho_reopen(t):
    if t < 500:
        return 2.0
    if t >= 1500:
        return RHO
    return 2.0 * (RHO / 2.0) ** ((t - 500) / 1000)


def train(arm, seed, iters=3000, lr=1e-2):
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    if arm.startswith("pen"):
        model = mb.build("pint", seed)
        lam = float(arm[3:])
    else:
        model = as_free(load_p(seed))     # θ := W_P (même modèle, r = ρ) : relâcher ρ ne fait pas sauter W
        lam = 0.0
    m = model.m
    m.__class__ = m2q.ExactProjBlock
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, log, status, t0 = [], [], "ok", time.time()
    for it in range(iters + 1):
        m.rho = float(torch.tensor(rho_reopen(it) if arm == "reopen" else RHO, dtype=torch.float32))
        out = model(u_tr)
        loss = ((out - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        W = m.weights()
        bc = mb.parts(m.H, m.k, m.nq, W["A"], W["B"], m.C)[2].mean()
        losses.append(float(loss.detach()))
        if it % 50 == 0:
            log.append(dict(it=it, loss=losses[-1], rho=m.rho, rW=float(m2q.r_W(m).max()), BC=float(bc)))
        if it == iters:
            break
        opt.zero_grad()
        (loss + lam * bc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    return dict(arm=arm, seed=seed, status=status, test=test, train=losses[-1], rW_end=float(m2q.r_W(m).max()),
                time=time.time() - t0, log=log, losses=losses)


def _job(a):
    r = train(*a)
    json.dump(r, open(os.path.join(OUT, f"{r['arm']}_s{r['seed']}.json"), "w"))
    return (f"train {r['arm']:8s} s{r['seed']}: {r['status']} train={r['train']:.4f} test={r['test']:.4f} "
            f"rW_end={r['rW_end']:.4f} {r['time']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("u1", "u2", "train"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "u1":
        res = [u1(s) for s in seeds]
        json.dump(res, open(os.path.join(OUT, "u1.json"), "w"), indent=1)
        for r in res:
            print(f"u1 s{r['seed']}: ρ_g (tangent) = {r['rho_g']:.3e} | ‖g‖/‖g0‖ = {r['rho_full']:.3e} | têtes saturées "
                  f"{r['n_sat']} | pente = {r['pente']:.4f} ({r['L2500']:.4f} → {r['L3000']:.4f})", flush=True)
    elif a.what == "u2":
        res = [u2(s) for s in seeds]
        json.dump(res, open(os.path.join(OUT, "u2.json"), "w"), indent=1)
        for r in res:
            print(f"u2 s{r['seed']}: perm {r['perm']} | IO {r['io_err']:.1e} | r {r['r_before']:.5f} → {r['r_after']:.5f} | "
                  "L(λ) : " + " ".join(f"{x:.3g}" for x in r["L"]), flush=True)
    else:
        jobs = [(ar, s, a.iters) for s, ar in itertools.product(seeds, a.arms.split(","))]
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
            for line in pool.imap_unordered(_job, jobs):
                print(line, flush=True)
