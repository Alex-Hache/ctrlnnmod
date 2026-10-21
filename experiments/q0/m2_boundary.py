"""
Q0 — mesure M2 : bord. Run libre des équations blocs depuis le même W₀ que `blockp`.

Bras (16 états, 8 têtes de 2 à porte, ρ = 0,999) :
  psat : `blockp`, init par défaut (r brut ≈ 5 > ρ) : W₀ = Φ(θ₀) est sur ∂S ;
  free : mêmes équations, s ≡ 1 (paramètres = W), init W₀ = Φ(θ₀) de psat ;
  pint : `blockp`, (Ã_h, B̃_h) × LAM·ρ/r_h : r_h = LAM·ρ, Φ = identité, W₀' = LAM·W₀ ;
  ren  : témoin.
Optimiseurs : adam, gd (SGD sans moment). Clip 10 dans les deux cas, comme `oscillator.run`.
Journal : perte, marge m(t) = ρ − maxₕ rₕ(W), têtes sur ∂S ou hors de S, r brut max ; test à la fin.
Critères : `analysis/m2.md` (écrits avant calcul).

Usage :
  PYTHONPATH=. python experiments/q0/m2_boundary.py --check
  PYTHONPATH=. python experiments/q0/m2_boundary.py --arms free,pint --opts adam --lrs 3e-3,1e-2 --seeds 0 --iters 500
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
sys.path.insert(0, os.path.join(HERE, "..", "memory"))
import oscillator as osc  # noqa: E402
from ctrlnmod.models.discrete.linf_gated import BlockGatedRNN  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2"))
RHO, LAM, TOL, LOG_EVERY = 0.999, 0.9, 1e-5, 50


class FreeBlockGatedRNN(BlockGatedRNN):
    """Mêmes équations, sans contrainte : W = (A_, B_) directement (s ≡ 1)."""

    def weights(self) -> dict:
        return dict(A=self.A_, B=self.B_)


def rows(m, A, B):
    """r_h(W) : Σₖ‖A_hk‖₂ + Σ_q‖B_hq‖₂ Σₖ‖C_qk‖₂ (même formule que BlockGatedRNN._block_rows)."""
    H, k = m.H, m.k
    nA = torch.linalg.matrix_norm(A.view(H, k, H, k).permute(0, 2, 1, 3), ord=2)
    nB = B.view(H, k, m.nq).norm(dim=1)
    c = m.C.view(m.nq, H, k).norm(dim=2).sum(1)
    return nA.sum(1) + nB @ c


@torch.no_grad()
def geometry(model):
    """Marge du certificat en W, têtes sur ∂S, têtes hors de S, r brut max."""
    m = model.m
    W = m.weights()
    r = rows(m, W["A"], W["B"])
    raw = m._block_rows()
    return dict(margin=float(RHO - r.max()), n_bd=int(((r - RHO).abs() < TOL).sum()),
                n_out=int((r > RHO + TOL).sum()), rmax_raw=float(raw.max()))


def build(arm, seed):
    torch.manual_seed(seed)
    if arm == "ren":
        return osc.make("ren")
    model = osc.make("blockp")                                  # θ₀ identique pour les trois bras
    m = model.m
    assert m.norm == "proj" and m.rho == RHO and m.ungated == 0
    if arm == "psat":
        return model
    with torch.no_grad():
        if arm == "free":
            W0 = {k: v.detach().clone() for k, v in m.weights().items()}
            m.__class__ = FreeBlockGatedRNN
            m.A_.copy_(W0["A"])
            m.B_.copy_(W0["B"])
        elif arm == "pint":
            r = m._block_rows()
            f = torch.clamp(LAM * RHO / r, max=1.0).repeat_interleave(m.k)[:, None]
            m.A_.mul_(f)
            m.B_.mul_(f)
        else:
            raise ValueError(arm)
    return model


def data(seed):
    A, B, C = osc.teacher(seed)
    u_tr, u_te = osc.inputs(256, 100, 10 + seed), osc.inputs(128, 300, 20 + seed)
    y_tr, y_te = osc.sim(A, B, C, u_tr), osc.sim(A, B, C, u_te)
    sc = y_tr.std()
    return u_tr, y_tr / sc, u_te, y_te / sc


def run(arm, opt_name, lr, seed, iters):
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = data(seed)
    model = build(arm, seed)
    block = arm != "ren"
    opt = (torch.optim.Adam if opt_name == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    log, status, t0 = [], "ok", time.time()
    first_contact, n_boundary_iters, margin_min, margin_max = None, 0, math.inf, -math.inf
    for it in range(iters + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        geo = geometry(model) if block else {}
        if block:
            on_or_out = geo["margin"] < TOL
            n_boundary_iters += on_or_out
            margin_min, margin_max = min(margin_min, geo["margin"]), max(margin_max, geo["margin"])
            if on_or_out and first_contact is None:
                first_contact = it
        if it % LOG_EVERY == 0:
            log.append(dict(it=it, loss=float(loss.detach()), **geo))
        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    return dict(arm=arm, opt=opt_name, lr=lr, seed=seed, iters=iters, status=status, test=test,
                train=log[-1]["loss"] if log else None, log=log, time=time.time() - t0,
                first_contact=first_contact, frac_boundary=n_boundary_iters / (iters + 1) if block else None,
                margin_min=margin_min if block else None, margin_max=margin_max if block else None)


def _job(a):
    arm, opt_name, lr, seed, iters = a
    r = run(*a)
    json.dump(r, open(os.path.join(OUT, f"m2_{arm}_{opt_name}_lr{lr:g}_s{seed}_it{iters}.json"), "w"), indent=1)
    fc = "" if r["first_contact"] is None else f" contact@{r['first_contact']}"
    mg = "" if r["margin_min"] is None else f" marge∈[{r['margin_min']:.3g}, {r['margin_max']:.3g}]"
    return (f"{arm:4s} {opt_name:4s} lr={lr:g} s{seed}: {r['status']} test={r['test']:.4f} train={r['train']:.4f}"
            f"{mg}{fc} {r['time']:.0f}s")


def check(seed=0):
    """c1 : perte(free, W₀) = perte(psat, θ₀). c2 : marges initiales ; Φ(θ₀') = θ₀' pour pint."""
    torch.set_num_threads(1)
    u_tr, y_tr, _, _ = data(seed)
    ms = {a: build(a, seed) for a in ("psat", "free", "pint")}
    with torch.no_grad():
        L = {a: float(((m(u_tr) - y_tr) ** 2).mean()) for a, m in ms.items()}
        for a, m in ms.items():
            print(a, f"perte={L[a]:.6f}", geometry(m))
        Wp, Wf = ms["psat"].m.weights(), ms["free"].m.weights()
        Wi = ms["pint"].m.weights()
        print("c1 écart relatif perte free/psat :", abs(L["free"] - L["psat"]) / L["psat"])
        print("c1 W free = W psat :", max(float((Wf[k] - Wp[k]).abs().max()) for k in "AB"))
        print("c2 Φ(θ₀') − θ₀' (pint) :", max(float((Wi[k] - getattr(ms["pint"].m, k + "_")).abs().max()) for k in "AB"))
        print("c2 W₀' − LAM·W₀ :", max(float((Wi[k] - LAM * Wp[k]).abs().max()) for k in "AB"))
        print("c2 marge attendue pint :", (1 - LAM) * RHO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arms", default="psat,free,pint,ren")
    ap.add_argument("--opts", default="adam,gd")
    ap.add_argument("--lrs", default=None, help="grille commune ; sinon --lrmap")
    ap.add_argument("--lrmap", default=None, help="JSON {\"arm/opt\": lr}")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    if a.check:
        check()
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    arms, opts, seeds = a.arms.split(","), a.opts.split(","), [int(s) for s in a.seeds.split(",")]
    if a.lrmap:
        lrmap = json.loads(a.lrmap)
        jobs = [(ar, o, lrmap[f"{ar}/{o}"], s, a.iters) for s in seeds for ar in arms for o in opts]
    else:
        lrs = [float(x) for x in a.lrs.split(",")]
        jobs = [(ar, o, lr, s, a.iters) for s, ar, o, lr in itertools.product(seeds, arms, opts, lrs)]
    jobs.sort(key=lambda j: j[0] == "ren")                       # blocs d'abord (plus lents)
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in pool.imap_unordered(_job, jobs):
            print(line, flush=True)
