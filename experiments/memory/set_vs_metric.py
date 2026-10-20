"""Ensemble ou métrique ? Test des oscillateurs (même maître, mêmes données, même initialisation que
oscillator.py). Trois élèves de mêmes équations :
  ren       : paramètres θ du REN, Adam sur θ          → ensemble contraint + métrique induite (JJᵀ)
  free      : poids W libres, Adam sur W                → pas d'ensemble, métrique euclidienne
  ren_eucl  : Adam sur W (pas euclidien ΔW), puis δθ = argmin ‖J δθ − ΔW‖² + λ‖δθ‖² (gradient
              conjugué, jvp/vjp) → reste dans l'ensemble REN, suit la métrique euclidienne des poids.
Si ren_eucl ≈ ren : l'ensemble compte. Si ren_eucl ≈ free : la métrique compte.
Usage : python experiments/memory/set_vs_metric.py  (ren et free : résultats de oscillator.py)"""
import json
import os
import sys
import time
from multiprocessing import Pool

import torch
from torch.func import functional_call, jvp, vjp

sys.path.insert(0, os.path.dirname(__file__))
from oscillator import inputs, sim, teacher  # noqa: E402

OUT = "experiments/memory/results_svm"
WK = ("A", "Bw", "Bu", "C1", "Lambda")          # poids issus du frame
TK = ("X", "Y", "B2")                           # paramètres qui les produisent


def free_forward(W, P, u):
    B, T, _ = u.shape
    x, ys = u.new_zeros(B, W["A"].shape[0]), []
    for t in range(T):
        w = torch.tanh((x @ W["C1"].T + u[:, t] @ P["D12"].T + P["bv"]) / W["Lambda"])
        ys.append(x @ P["C2"].T + w @ P["D21"].T + u[:, t] @ P["D22"].T + P["by"])
        x = x @ W["A"].T + w @ W["Bw"].T + u[:, t] @ W["Bu"].T + P["bx"]
    return torch.stack(ys, 1)


def run(a):
    lr, seed, iters, lam_reg, cg_it = a
    torch.set_num_threads(1)
    from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
    torch.manual_seed(seed)
    A, B, C = teacher(seed)
    u_tr, u_te = inputs(256, 100, 10 + seed), inputs(128, 300, 20 + seed)
    y_tr, y_te = sim(A, B, C, u_tr), sim(A, B, C, u_te)
    sc = y_tr.std()
    y_tr, y_te = y_tr / sc, y_te / sc
    ren = ExplicitREN(2, 2, 16, 32, gamma=None, alpha_bar=1.0)      # même graine → même init que "ren"
    theta = {k: v.detach().clone() for k, v in ren.named_parameters() if k in TK}
    P = {k: torch.nn.Parameter(v.detach().clone()) for k, v in ren.named_parameters() if k not in TK}

    def frame(th):
        fr = functional_call(ren, {**th, **{k: v.detach() for k, v in P.items()}}, (), {}, strict=False) \
            if False else None
        return fr

    class F(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.r = ren

        def forward(s):
            fr = s.r.frame()
            return tuple(fr[k] for k in WK)
    fmod = F()

    def wmap(*tv):
        return functional_call(fmod, {"r." + k: v for k, v in zip(TK, tv)}, ())

    W0 = wmap(*[theta[k] for k in TK])
    S = {k: torch.nn.Parameter(v.detach().clone()) for k, v in zip(WK, W0)}     # poids « ombre »
    opt = torch.optim.Adam(list(S.values()) + list(P.values()), lr=lr)
    t0, status, log, fid = time.time(), "ok", [], []
    for it in range(iters):
        tv = [theta[k] for k in TK]
        Wc = wmap(*tv)
        with torch.no_grad():
            for k, v in zip(WK, Wc):
                S[k].copy_(v)
        loss = ((free_forward(S, P, u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(S.values()) + list(P.values()), 10.0)
        opt.step()                                                   # pas euclidien sur W (et P)
        dW = [S[k].detach() - v.detach() for k, v in zip(WK, Wc)]
        # δθ = Jᵀ (JJᵀ + λ)⁻¹ ΔW, gradient conjugué dans l'espace des poids
        _, vjp_fn = vjp(wmap, *tv)

        def JJt(v):
            g = vjp_fn(tuple(v))
            return [o + lam_reg * vi for o, vi in zip(jvp(wmap, tuple(tv), tuple(g))[1], v)]
        dot = lambda a_, b_: sum((x * y).sum() for x, y in zip(a_, b_))  # noqa: E731
        z = [torch.zeros_like(d) for d in dW]
        r = [d.clone() for d in dW]
        p = [d.clone() for d in dW]
        rr = dot(r, r)
        for _ in range(cg_it):
            Ap = JJt(p)
            al = rr / dot(p, Ap).clamp(min=1e-30)
            z = [zi + al * pi for zi, pi in zip(z, p)]
            r = [ri - al * ai for ri, ai in zip(r, Ap)]
            rn = dot(r, r)
            if rn < 1e-12 * dot(dW, dW):
                break
            p = [ri + (rn / rr) * pi for ri, pi in zip(r, p)]
            rr = rn
        dth = vjp_fn(tuple(z))
        Jd = jvp(wmap, tuple(tv), tuple(dth))[1]
        with torch.no_grad():
            for k, d in zip(TK, dth):
                theta[k] = theta[k] + d
        if it % 50 == 0:
            log.append(float(loss.detach()))
            fid.append(float(torch.sqrt(dot([a_ - b_ for a_, b_ in zip(Jd, dW)], [a_ - b_ for a_, b_ in zip(Jd, dW)])
                                        / dot(dW, dW).clamp(min=1e-30))))
    with torch.no_grad():
        S2 = dict(zip(WK, wmap(*[theta[k] for k in TK])))
        test = float(((free_forward(S2, P, u_te) - y_te) ** 2).mean())
        rho = float(torch.linalg.eigvals(S2["A"]).abs().max())
    res = dict(kind="ren_eucl", lr=lr, seed=seed, status=status, test=test, rho_A=rho,
               train=log[-1] if log else None, log=log, step_misfit=fid, time=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/svm_s{seed}_lr{lr}.json", "w"))
    return res


if __name__ == "__main__":
    iters = int(os.environ.get("ITERS", 2000))
    jobs = [(lr, s, iters, 1e-6, 30) for s in range(3) for lr in (1e-3, 3e-3, 1e-2)]
    if os.environ.get("SMOKE"):
        jobs = [(3e-3, 0, 20, 1e-6, 30)]
    with Pool(min(9, len(jobs))) as pool:
        for r in pool.imap_unordered(run, jobs):
            print(r["seed"], r["lr"], r["status"], f"test {r['test']:.4f}", f"rho {r['rho_A']:.3f}",
                  f"misfit {r['step_misfit'][-1] if r['step_misfit'] else float('nan'):.2e}", f"{r['time']:.0f}s", flush=True)
