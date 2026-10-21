"""
Q0 — M2-q : tests bon marché sur les blocs de 2. Critères : `analysis/m2q.md` (poussés avant calcul, 56ca82b).

  t1 : par quel bloc passe la descente (GD η = 0,03 depuis W₀', jusqu'au contact) ;
  t4 : bras free, p, pgd, fix2, alent, arapide (Adam lr 1e-2, 3000 itér., init W₀') ; pgd = T2 ;
  t5 : contraction empirique des modèles finaux de t4 (float64).
(T3 utilise `m2bis.py --opt gd --betas ...`.)

Usage :
  PYTHONPATH=. python experiments/q0/m2q.py t1
  PYTHONPATH=. python experiments/q0/m2q.py t4 --seeds 0,1,2,3,4
  PYTHONPATH=. python experiments/q0/m2q.py t5
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402
import m2bis as mb  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2q"))
RHO, TOL = m2.RHO, m2.TOL
ARMS = ("free", "p", "pgd", "fix2", "alent", "arapide")
GROUP = {"m.A_": "Ã", "m.B_": "B̃", "m.C": "C", "m.D": "D", "m.E": "E", "m.b": "biais", "m.bx": "biais",
         "m.gate_proj.weight": "portes", "m.gate_proj.bias": "portes", "m.c": "c (inutilisé)",
         "C.weight": "lecture", "C.bias": "lecture"}


# ------------------------------------------------------------------ T1
def t1(seed, eta=0.03, iters=5000):
    torch.set_num_threads(1)
    u, y, _, _ = m2.data(seed)
    model = mb.build("pint", seed)
    m = model.m
    H, k, nq = m.H, m.k, m.nq
    P = dict(m.named_parameters())
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    steps, tc = [], None
    for it in range(iters + 1):
        loss = ((model(u) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            r = mb.r_raw(H, k, nq, P["A_"], P["B_"], P["C"])
        if float(r.max()) >= RHO - TOL:
            tc = it
            break
        if it == iters:
            break
        g = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        tot = sum(float((x ** 2).sum()) for x in g.values())
        share = {}
        for n, x in g.items():
            share[GROUP[n]] = share.get(GROUP[n], 0.0) + float((x ** 2).sum()) / tot
        v = {X: float(g["m." + X].norm() / P[X].detach().norm()) for X in mb.RP}
        hs = int(r.argmax())
        d, o, b = mb.parts(H, k, nq, P["A_"], P["B_"], P["C"])
        dpart = {}
        for lab, part in (("Ad", d[hs]), ("Ao", o[hs]), ("BC", b[hs])):
            gr = torch.autograd.grad(part, [P[X] for X in mb.RP], retain_graph=True)
            dpart[lab] = float(-sum((a * g["m." + X]).sum() for a, X in zip(gr, mb.RP)))
        steps.append(dict(it=it, loss=float(loss.detach()), share=share, v=v, dpart=dpart, h=hs))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    return dict(seed=seed, eta=eta, tc=tc, L0=steps[0]["loss"] if steps else None, steps=steps)


# ------------------------------------------------------------------ T2 / T4
def rho_t(arm, t):
    if arm == "free":
        return math.inf
    if arm in ("p", "pgd"):
        return RHO
    if arm == "fix2":
        return 2.0
    t0, t1_ = {"alent": (500, 2500), "arapide": (1500, 2000)}[arm]
    if t < t0:
        return 100.0
    if t >= t1_:
        return RHO
    return 100.0 * (RHO / 100.0) ** ((t - t0) / (t1_ - t0))


def build4(arm, seed):
    model = mb.build("pint", seed)                         # W₀' (0,9ρ) ; Φ = identité dans S
    if arm in ("free", "pgd"):
        model.m.__class__ = m2.FreeBlockGatedRNN
    return model


def r_W(m):
    with torch.no_grad():
        W = m.weights()
        return m2.rows(m, W["A"], W["B"])


def t4(arm, seed, iters=3000, lr=1e-2):
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = build4(arm, seed)
    m = model.m
    H, k, nq = m.H, m.k, m.nq
    proj = arm not in ("free", "pgd")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, rraw, log, status, t0 = [], [], [], "ok", time.time()
    for it in range(iters + 1):
        rt = rho_t(arm, it)
        if proj:
            m.rho = rt
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        with torch.no_grad():
            raw = mb.r_raw(H, k, nq, m.A_, m.B_, m.C)
        losses.append(float(loss.detach()))
        rraw.append(float(raw.max()))
        if it % 50 == 0:
            log.append(dict(it=it, loss=losses[-1], rho_t=rt if math.isfinite(rt) else None,
                            rW=float(r_W(m).max()), rraw=rraw[-1],
                            n_sat=int((raw > rt + TOL).sum()) if proj else None))
        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if arm == "pgd":                                    # rétraction radiale : r_h ≤ ρ exactement
            with torch.no_grad():
                r = mb.r_raw(H, k, nq, m.A_, m.B_, m.C)
                f = torch.clamp(RHO / r, max=1.0).repeat_interleave(k)[:, None]
                m.A_.mul_(f)
                m.B_.mul_(f)
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"t4_{arm}_s{seed}.pt"))
    return dict(arm=arm, seed=seed, lr=lr, iters=iters, status=status, test=test, train=losses[-1],
                rW_end=float(r_W(m).max()), rho_end=rho_t(arm, iters) if arm != "free" else None,
                time=time.time() - t0, log=log, losses=losses, rraw=rraw)


def _job4(a):
    r = t4(*a)
    json.dump(r, open(os.path.join(OUT, f"t4_{r['arm']}_s{r['seed']}.json"), "w"))
    return (f"t4 {r['arm']:7s} s{r['seed']}: {r['status']} train={r['train']:.4f} test={r['test']:.4f} "
            f"rW_end={r['rW_end']:.3f} {r['time']:.0f}s")


# ------------------------------------------------------------------ T5
@torch.no_grad()
def t5(arm, seed, n=32, T=2000, pairs=4):
    torch.set_num_threads(1)
    model = build4(arm, seed)
    model.load_state_dict(torch.load(os.path.join(OUT, f"t4_{arm}_s{seed}.pt")))
    model = model.double()
    m = model.m
    if arm != "free" and arm != "pgd":
        m.rho = rho_t(arm, 3000)
    u = m2.osc.inputs(n, T, 30 + seed).double().repeat(pairs, 1, 1)
    g = torch.Generator().manual_seed(77 + seed)
    out = {}
    for scale in ("grande", "petite"):
        x0 = torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64)
        x1 = (torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64) if scale == "grande"
              else x0 + 1e-3 * torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64))

        def dist(a, b):
            return (a - b).view(*a.shape[:-1], m.H, m.k).norm(dim=-1).max(-1).values

        d0 = dist(x0, x1)                                    # (N,)
        xa, xb = m.forward(u, x0), m.forward(u, x1)          # (N, T, nx)
        ratio = dist(xa, xb) / d0[:, None]                   # (N, T)
        out[scale] = dict(E=float(ratio.max()), final_med=float(ratio[:, -1].median()),
                          final_max=float(ratio[:, -1].max()))
    out["E"] = max(out["grande"]["E"], out["petite"]["E"])
    return dict(arm=arm, seed=seed, rW_end=float(r_W(m).max()), **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("t1", "t4", "t5"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "t1":
        res = [t1(s) for s in seeds]
        json.dump(res, open(os.path.join(OUT, "t1.json"), "w"))
        for r in res:
            print(f"t1 s{r['seed']}: t_c = {r['tc']}, pas avant contact = {len(r['steps'])}")
    elif a.what == "t4":
        jobs = [(ar, s, a.iters) for s, ar in itertools.product(seeds, a.arms.split(","))]
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
            for line in pool.imap_unordered(_job4, jobs):
                print(line, flush=True)
    else:
        res = [t5(ar, s) for ar in a.arms.split(",") for s in seeds]
        json.dump(res, open(os.path.join(OUT, "t5.json"), "w"))
        for r in res:
            print(f"t5 {r['arm']:7s} s{r['seed']}: E={r['E']:.4g} | d_T/d0 méd. grande {r['grande']['final_med']:.3g}, "
                  f"petite {r['petite']['final_med']:.3g} | rW_end={r['rW_end']:.3f}")
