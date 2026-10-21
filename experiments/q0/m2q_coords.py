"""Q0 M2-q T6 — le libre est-il équivalent en entrée-sortie à un point de S ?
Changement de coordonnées x' = T x, T bloc-diagonal (2×2 par tête). Critères : `analysis/m2q.md` § T6 (3193e98).
S1 : T = diag(α_h I₂), convexe en log α. S2 : T_h général, optimisation locale.
Usage : PYTHONPATH=. python experiments/q0/m2q_coords.py"""
import copy
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402
import m2q  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
RHO, D = m2.RHO, os.path.join(HERE, "results_m2q")


def load_free(seed):
    model = m2q.build4("free", seed)
    model.load_state_dict(torch.load(os.path.join(D, f"t4_free_s{seed}.pt")))
    return model.double()


def r_of(m, T):
    """r_h après x' = T x ; T : (H, k, k)."""
    H, k, nq = m.H, m.k, m.nq
    Ti = torch.linalg.inv(T)
    A = m.A_.view(H, k, H, k).permute(0, 2, 1, 3)                       # (H, H, k, k) blocs A_hk
    A2 = T[:, None] @ A @ Ti[None, :]                                   # T_h A_hk T_k⁻¹
    nA = torch.linalg.matrix_norm(A2, ord=2).sum(1)
    B = m.B_.view(H, k, nq)                                             # B_h : (k, nq)
    nB = (T @ B).norm(dim=1)                                            # (H, nq)
    C = m.C.view(nq, H, k)                                              # C_qk : (1, k)
    c = (C[:, :, None, :] @ Ti[None]).squeeze(2).norm(dim=-1).sum(1)    # (nq,)
    return nA + nB @ c


def s1(m):
    """min_α maxₕ r_h, posynôme : lissage log-sum-exp avec continuation, LBFGS en log α."""
    H, k, nq = m.H, m.k, m.nq
    with torch.no_grad():
        a = torch.linalg.matrix_norm(m.A_.view(H, k, H, k).permute(0, 2, 1, 3), ord=2)   # (H, H)
        b = m.B_.view(H, k, nq).norm(dim=1)                                               # (H, nq)
        c = m.C.view(nq, H, k).norm(dim=2)                                                # (nq, H)
    y = torch.zeros(H, requires_grad=True)

    def r(y):
        al = torch.exp(y)
        return (a * al[:, None] / al[None, :]).sum(1) + al * (b @ (c / al[None, :]).sum(1))

    for K in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        opt = torch.optim.LBFGS([y], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            f = torch.logsumexp(K * torch.log(r(y)), 0) / K
            f.backward()
            return f
        opt.step(closure)
    with torch.no_grad():
        al = torch.exp(y - y[0])
        return al, float(r(y).max())


def s2(m, al, starts=5, steps=3000):
    H, k = m.H, m.k
    best, bestT = float("inf"), None
    g = torch.Generator().manual_seed(0)
    for st in range(starts):
        T0 = al[:, None, None] * torch.eye(k)[None]
        if st:
            T0 = T0 * (1 + 0.1 * torch.randn(H, k, k, generator=g))
        T = T0.clone().requires_grad_(True)
        opt = torch.optim.Adam([T], lr=1e-2 * float(al.mean()))
        for it in range(steps):
            K = 10.0 if it < steps // 3 else 100.0 if it < 2 * steps // 3 else 1000.0
            opt.zero_grad()
            f = torch.logsumexp(K * torch.log(r_of(m, T)), 0) / K
            f.backward()
            opt.step()
        with torch.no_grad():
            v = float(r_of(m, T).max())
            cond = float(torch.linalg.cond(T).max())
        if v < best:
            best, bestT, bestcond = v, T.detach().clone(), cond
    return bestT, best, bestcond


@torch.no_grad()
def transform(model, T):
    """Copie du modèle en coordonnées x' = T x (T bloc-diagonal)."""
    new = copy.deepcopy(model)
    m, n = new.m, new.m.nx
    Tf = torch.block_diag(*T)
    Ti = torch.linalg.inv(Tf)
    m.A_.copy_(Tf @ m.A_ @ Ti)
    m.B_.copy_(Tf @ m.B_)
    m.C.copy_(m.C @ Ti)
    m.E.copy_(Tf @ m.E)
    m.bx.copy_(Tf @ m.bx)
    new.C.weight.copy_(new.C.weight @ Ti)
    return new


res, lines = [], []
for seed in range(5):
    model = load_free(seed)
    m = model.m
    stored = json.load(open(os.path.join(D, f"t4_free_s{seed}.json")))["rW_end"]
    I = torch.eye(m.k)[None].repeat(m.H, 1, 1)
    with torch.no_grad():
        r1 = float(r_of(m, I).max())
    al, rs1 = s1(m)
    T1 = al[:, None, None] * torch.eye(m.k)[None]
    with torch.no_grad():
        rs1_check = float(r_of(m, T1).max())
    T2, rs2, cond2 = s2(m, al)
    _, _, u_te, _ = m2.data(seed)
    u_te = u_te.double()
    with torch.no_grad():
        y0 = model(u_te)
        err = {lab: float((transform(model, T)(u_te) - y0).abs().max() / y0.abs().max())
               for lab, T in (("S1", T1), ("S2", T2))}
    rec = dict(seed=seed, r_identity=r1, r_stored=stored, c_r=abs(r1 - stored) / stored < 1e-5,
               r_S1=rs1, r_S1_check=rs1_check, alpha=[float(x) for x in al], r_S2=rs2, cond_T_S2=cond2,
               io_err=err, c_io=all(e < 1e-9 for e in err.values()))
    res.append(rec)
    line = (f"graine {seed} : r(T=I) = {r1:.3f} (stocké {stored:.3f}) | r*(S1) = {rs1:.4f} (α : "
            + " ".join(f"{x:.3g}" for x in al) + f") | r*(S2) = {rs2:.4f} (cond T max {cond2:.3g}) | "
            f"erreur IO relative S1 {err['S1']:.1e}, S2 {err['S2']:.1e}")
    print(line, flush=True)
    lines.append(line)

ok = all(r["c_r"] and r["c_io"] for r in res)
n1 = sum(r["r_S1"] <= RHO for r in res)
n2 = sum(r["r_S2"] <= RHO for r in res)
out = [f"Contrôles c-r et c-IO : {'passés' if ok else 'ÉCHEC → pas de lecture'}",
       f"r*(S1) ≤ ρ sur {n1}/5 ; r*(S2) ≤ ρ sur {n2}/5 ; r*(S1)/ρ = " + " ".join(f"{r['r_S1'] / RHO:.3g}" for r in res)
       + " ; r*(S2)/ρ = " + " ".join(f"{r['r_S2'] / RHO:.3g}" for r in res)]
if ok:
    if n1 >= 3:
        out.append("Ligne T6 : le libre est exactement équivalent en entrée-sortie à un modèle de S (mise à l'échelle par tête) : "
                   "l'obstacle est le système de coordonnées du certificat, pas S comme famille de fonctions")
    elif n2 >= 3:
        out.append("Ligne T6 : même lecture, avec un changement de base général par tête (S2)")
    else:
        out.append("Ligne T6 : les changements de coordonnées par tête ne suffisent pas (directions peu observables, borne "
                   "grossière sur le chemin tanh, ou modèle hors de toute version de S)")
for o in out:
    print(o)
json.dump(dict(res=res, lecture=out), open(os.path.join(D, "t6.json"), "w"), indent=1)
open(os.path.join(HERE, "m2q_T6_analysis.txt"), "w").write("\n".join(lines + out) + "\n")
