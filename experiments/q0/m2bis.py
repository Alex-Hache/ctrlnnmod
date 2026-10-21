"""
Q0 — M2-bis : P-int et libre depuis la même init intérieure W₀', à petit pas d'Adam.
Critères : `analysis/m2bis.md` (écrits et poussés avant calcul, 4c59261).

Bras :
  pint    : `blockp`, (Ã_h, B̃_h) × 0,9ρ/r_h (comme M2) ; W₀' = Φ(θ₀') = θ₀' ;
  freeint : mêmes équations, s ≡ 1, init W₀'.
Jusqu'au premier contact t_c (inclus), à chaque pas :
  témoin d'échelle : r_max(θ₀' + σ_j ⊙ Δ(t)), Δ(t) = θ_t − θ₀' sur (Ã, B̃, C), 20 masques σ_j fixes ;
  direction du flot GD : g_GD = ⟨∇r_h*, −∇L⟩ / ‖∇L‖ sur (Ã, B̃, C), h* = tête de r max.

Usage :
  PYTHONPATH=. python experiments/q0/m2bis.py --check
  PYTHONPATH=. python experiments/q0/m2bis.py --arms pint,freeint --lrs 1e-3,3e-4,1e-4 --seeds 0 --iters 300
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2bis"))
RHO, TOL, NMASK, LOG_EVERY = m2.RHO, m2.TOL, 20, 10
RP = ("A_", "B_", "C")                                  # paramètres dont dépend r


def build(arm, seed):
    model = m2.build("pint", seed)
    if arm == "freeint":
        model.m.__class__ = m2.FreeBlockGatedRNN           # Φ(θ₀') = θ₀' : W₀' = (Ã, B̃)
    elif arm != "pint":
        raise ValueError(arm)
    return model


def r_raw(H, k, nq, A, B, C):
    """r_h brut à partir de (Ã, B̃, C), même formule que `_block_rows`, avec dimensions de tête (lot)."""
    lead = A.shape[:-2]
    nA = torch.linalg.matrix_norm(A.reshape(*lead, H, k, H, k).transpose(-3, -2), ord=2)   # (..., H, H)
    nB = B.reshape(*lead, H, k, nq).norm(dim=-2)                                          # (..., H, nq)
    c = C.reshape(*lead, nq, H, k).norm(dim=-1).sum(-1)                                   # (..., nq)
    return nA.sum(-1) + (nB * c[..., None, :]).sum(-1)                                    # (..., H)


def parts(H, k, nq, A, B, C):
    """(part Ã diagonal, part Ã hors diagonale, part B̃·C) de r_h, par tête."""
    nA = torch.linalg.matrix_norm(A.reshape(H, k, H, k).transpose(1, 2), ord=2)          # (H, H)
    d = torch.diagonal(nA)
    c = C.reshape(nq, H, k).norm(dim=2).sum(1)
    return d, nA.sum(1) - d, B.reshape(H, k, nq).norm(dim=1) @ c


@torch.no_grad()
def contact_diag(H, k, nq, th0, P, masks, hc):
    """Diagnostic au contact (M2-ter, amendement) : bloc moteur X*, mode échelle / orientation, parts."""
    r0 = float(r_raw(H, k, nq, *(th0[n] for n in RP)).max())
    blocks = {}
    for X in RP:
        real = float(r_raw(H, k, nq, *(P[n] if n == X else th0[n] for n in RP)).max())
        mir = r_raw(H, k, nq, *((th0[n] + masks[n] * (P[n] - th0[n])) if n == X
                                 else th0[n].expand(NMASK, *th0[n].shape) for n in RP)).max(-1).values
        blocks[X] = dict(real=real, med=float(mir.median()), max=float(mir.max()))
    xs = max(RP, key=lambda X: blocks[X]["real"])
    d0, o0, b0 = parts(H, k, nq, *(th0[n] for n in RP))
    d1, o1, b1 = parts(H, k, nq, *(P[n] for n in RP))
    return dict(r0=r0, blocks=blocks, driver=xs,
                mode="orientation" if blocks[xs]["real"] > blocks[xs]["max"] else "échelle", head=hc,
                init=dict(Ad=float(d0[hc]), Ao=float(o0[hc]), BC=float(b0[hc])),
                contact=dict(Ad=float(d1[hc]), Ao=float(o1[hc]), BC=float(b1[hc])))


def run(arm, lr, seed, iters, beta=1.0, stop=False, opt_name="adam"):
    """beta : lr de B̃ = beta·lr (M2-ter). stop : arrêt au premier contact. opt_name : adam ou gd (sans moment)."""
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = build(arm, seed)
    m = model.m
    H, k, nq = m.H, m.k, m.nq
    P = dict(m.named_parameters())
    th0 = {n: P[n].detach().clone() for n in RP}
    g = torch.Generator().manual_seed(1000 + seed)       # mêmes masques pour pint et freeint
    masks = {n: (2 * torch.randint(0, 2, (NMASK, *th0[n].shape), generator=g) - 1).to(th0[n].dtype)
             for n in RP}
    others = [p for p in model.parameters() if p is not m.B_]
    groups = [dict(params=others, lr=lr), dict(params=[m.B_], lr=beta * lr)]
    opt = torch.optim.Adam(groups) if opt_name == "adam" else torch.optim.SGD(groups)
    log, tc, status, L0, t0, cdiag = [], None, "ok", None, time.time(), None
    for it in range(iters + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        L = float(loss.detach())
        L0 = L if L0 is None else L0
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            r = r_raw(H, k, nq, *(P[n] for n in RP))
        rmax = float(r.max())
        rec = dict(it=it, loss=L, rmax=rmax, n_on_or_out=int((r >= RHO - TOL).sum()))
        if tc is None:                                    # témoins jusqu'au contact inclus
            with torch.no_grad():
                rm = r_raw(H, k, nq, *(th0[n] + masks[n] * (P[n] - th0[n]) for n in RP)).max(-1).values
            hs = int(r.argmax())
            gr = torch.autograd.grad(r_raw(H, k, nq, *(P[n] for n in RP))[hs], [P[n] for n in RP])
            gl = [P[n].grad for n in RP]
            gnorm = torch.sqrt(sum((x ** 2).sum() for x in gl))
            ggd = float(-sum((a * b).sum() for a, b in zip(gr, gl)) / gnorm)
            with torch.no_grad():
                pd_, po_, pb_ = parts(H, k, nq, *(P[n] for n in RP))
            rec.update(q=float((rm >= RHO - TOL).float().mean()), rmask_min=float(rm.min()),
                       rmask_med=float(rm.median()), rmask_max=float(rm.max()), ggd=ggd, h=hs,
                       partAd=float(pd_[hs]), partAo=float(po_[hs]), partBC=float(pb_[hs]))
            if rmax >= RHO - TOL:
                tc = it
                cdiag = contact_diag(H, k, nq, th0, P, masks, hs)
            log.append(rec)
        elif it % LOG_EVERY == 0:
            log.append(rec)
        if it == iters or (stop and tc is not None):
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    by_it = {e["it"]: e for e in log}
    out = dict(arm=arm, opt=opt_name, lr=lr, beta=beta, stop=stop, seed=seed, iters=iters, status=status,
               test=test, L0=L0, train=log[-1]["loss"], tc=tc, time=time.time() - t0, contact=cdiag, log=log)
    if tc is not None:
        pre = [by_it[i]["ggd"] for i in range(max(0, tc - 10), tc)]
        out.update(L_pre=by_it[tc - 1]["loss"], q_tc=by_it[tc]["q"], ggd_pre=sum(pre) / len(pre),
                   rmask_tc=[by_it[tc][x] for x in ("rmask_min", "rmask_med", "rmask_max")])
    return out


def cause(r):
    """Cause du contact, règle de `analysis/m2bis.md`."""
    if r["tc"] is None:
        return "pas de contact"
    if r["q_tc"] >= 0.5:
        return "échelle"
    if r["q_tc"] == 0:
        return "gradient" if r["ggd_pre"] > 0 else "orientation Adam"
    return "mixte"


def _job(a):
    r = run(*a)
    arm, lr, seed, iters, beta, stop, opt_name = a
    tag = ("_gd" if opt_name == "gd" else "") + ("" if beta == 1 else f"_beta{beta:g}") + ("_stop" if stop else "")
    json.dump(r, open(os.path.join(OUT, f"m2bis_{arm}_lr{lr:g}{tag}_s{seed}_it{iters}.json"), "w"), indent=1)
    s = (f"{arm:7s} {opt_name} lr={lr:g} β={beta:g} s{seed}: {r['status']} L0={r['L0']:.4f} train={r['train']:.4f}"
         f" test={r['test']:.4f}")
    if r["contact"] is not None:
        c = r["contact"]
        s += f" | X*={c['driver']} {c['mode']} parts(Ãd,Ão,B̃C) {c['init']['Ad']:.3f},{c['init']['Ao']:.3f},{c['init']['BC']:.3f}"
        s += f" → {c['contact']['Ad']:.3f},{c['contact']['Ao']:.3f},{c['contact']['BC']:.3f}"
    if r["tc"] is None:
        s += f" | pas de contact, rmax fin={r['log'][-1]['rmax']:.3f}"
    else:
        s += (f" | t_c={r['tc']} L(t_c−1)/L0={r['L_pre'] / r['L0']:.3f} q={r['q_tc']:.2f}"
              f" rmask(min/méd/max)={'/'.join(f'{x:.3f}' for x in r['rmask_tc'])} g_GD={r['ggd_pre']:+.3f}"
              f" → {cause(r)}")
    return s + f" {r['time']:.0f}s"


def check(seed=0):
    """c0 : r_raw = _block_rows ; marge initiale 0,1ρ ; W₀' identique pour les deux bras."""
    torch.set_num_threads(1)
    ms = {a: build(a, seed) for a in ("pint", "freeint")}
    with torch.no_grad():
        for a, mod in ms.items():
            m = mod.m
            r = r_raw(m.H, m.k, m.nq, m.A_, m.B_, m.C)
            print(a, "c0 |r_raw − _block_rows| =", float((r - m._block_rows()).abs().max()),
                  "| marge =", RHO - float(r.max()), "(attendu", 0.1 * RHO, ")")
        Wp, Wf = ms["pint"].m.weights(), ms["freeint"].m.weights()
        print("W₀' pint − freeint :", max(float((Wp[x] - Wf[x]).abs().max()) for x in "AB"))
        u, y, _, _ = m2.data(seed)
        print("perte init pint, freeint :", *(float(((mod(u) - y) ** 2).mean()) for mod in ms.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arms", default="pint,freeint")
    ap.add_argument("--lrs", default="1e-3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--betas", default="1", help="lr de B̃ = β·lr (M2-ter)")
    ap.add_argument("--stop", action="store_true", help="arrêt au premier contact")
    ap.add_argument("--opt", default="adam", choices=("adam", "gd"))
    a = ap.parse_args()
    if a.check:
        check()
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ar, float(lr), int(s), a.iters, float(b), a.stop, a.opt) for s, ar, lr, b in
            itertools.product(a.seeds.split(","), a.arms.split(","), a.lrs.split(","), a.betas.split(","))]
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in pool.imap_unordered(_job, jobs):
            print(line, flush=True)
