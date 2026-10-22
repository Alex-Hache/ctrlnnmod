"""
Session « gradient non nul, perte plate » — test KKT relâché aux plis (piste 2).
Critère : `experiments/q0/sessions/theorie_gradient_plat.md` § 6 (poussé avant calcul, 462a1c8).

Au point final de P (T4, régénéré), W = Φ(θ). Pour δ ∈ {0, 1e-4, 1e-3, 1e-2} :
  R_δ = min ‖g_con + Σ_h λ_h ξ_h‖,  λ_h ≥ 0,  ξ_h ∈ ∂_δ r_h(W),
où chaque norme ‖X‖ de r_h a pour sous-différentiel relâché {G : ‖G‖_dual ≤ 1, ⟨G, X⟩ ≥ ‖X‖ − δ}.
Changement de variable Z = λ·G : problème SOCP (norme nucléaire 2×2 par deux cônes d'ordre 2).

Usage :
  REGEN=<dossier des .pt régénérés> PYTHONPATH=. python experiments/q0/gp_kkt.py
"""
import copy
import json
import os
import sys
import time

import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402
import m2q  # noqa: E402

REGEN = os.environ["REGEN"]
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_gp"))
RHO = m2.RHO
DELTAS = (0.0, 1e-4, 1e-3, 1e-2)
D_DEC = 1e-2
N_RAND = 20


def as_free(model):
    """Copie libre qui a exactement les poids effectifs W = Φ(θ) (comme m2t.as_free)."""
    new = copy.deepcopy(model)
    with torch.no_grad():
        W = model.m.weights()
        new.m.__class__ = m2.FreeBlockGatedRNN
        new.m.A_.copy_(W["A"])
        new.m.B_.copy_(W["B"])
    return new


def nuc2(Z):
    """Norme nucléaire d'une matrice 2×2 réelle : max(‖(a+d, c−b)‖, ‖(a−d, b+c)‖)."""
    a, b, c, d = Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1]
    return max(np.hypot(a + d, c - b), np.hypot(a - d, b + c))


def check_nuc2():
    rng = np.random.default_rng(0)
    for _ in range(200):
        Z = rng.standard_normal((2, 2))
        assert abs(nuc2(Z) - np.linalg.svd(Z, compute_uv=False).sum()) < 1e-12


class KKT:
    """Résidu KKT relâché pour un W fixé ; g entre comme un vecteur paramètre additif (témoins).
    Ordre des coordonnées (même ordre que `flat`) : blocs A (h, j, [a b c d]), B (h, q, [2]), C (q, j, [2])."""

    def __init__(self, A, B, C, sat, H, k, nq, delta):
        self.H, self.nq = H, nq
        Ab = A.reshape(H, 2, H, 2).transpose(0, 2, 1, 3).reshape(H * H, 2, 2)        # bloc (h, j)
        nA = np.linalg.norm(Ab, 2, axis=(1, 2))
        Bb = B.reshape(H, 2, nq).transpose(0, 2, 1).reshape(H * nq, 2)                 # (h, q)
        nB = np.linalg.norm(Bb, axis=1)
        Cb = C.reshape(nq, H, 2).reshape(nq * H, 2)                                    # (q, j)
        nC = np.linalg.norm(Cb, axis=1)
        cq = nC.reshape(nq, H).sum(1)
        RA = np.kron(np.eye(H), np.ones((H, 1)))       # λ_h répété sur j
        RB = np.kron(np.eye(H), np.ones((nq, 1)))      # λ_h répété sur q
        RC = np.kron(np.eye(nq), np.ones((H, 1)))      # μ_q répété sur j
        T1 = np.array([[1, 0, 0, 1], [0, -1, 1, 0]], float)
        T2 = np.array([[1, 0, 0, -1], [0, 1, 1, 0]], float)
        lam = cp.Variable(H, nonneg=True)
        Z = cp.Variable((H * H, 4))
        Y = cp.Variable((H * nq, 2))
        V = cp.Variable((nq * H, 2))
        lA, lB = RA @ lam, RB @ lam
        mu = RC @ (nB.reshape(H, nq).T @ lam)
        cons = [lam[h] == 0 for h in range(H) if h not in sat]
        cons += [cp.norm(Z @ T1.T, axis=1) <= lA, cp.norm(Z @ T2.T, axis=1) <= lA,
                 cp.sum(cp.multiply(Z, Ab.reshape(H * H, 4)), axis=1) >= cp.multiply(lA, nA - delta),
                 cp.norm(Y, axis=1) <= lB, cp.sum(cp.multiply(Y, Bb), axis=1) >= cp.multiply(lB, nB - delta),
                 cp.norm(V, axis=1) <= mu, cp.sum(cp.multiply(V, Cb), axis=1) >= cp.multiply(mu, nC - delta)]
        cqB = np.tile(cq, H)[:, None]
        lin = cp.hstack([cp.vec(Z, order="C"), cp.vec(cp.multiply(cqB, Y), order="C"), cp.vec(V, order="C")])
        self.g = cp.Parameter(H * H * 4 + H * nq * 2 + nq * H * 2)
        self.prob = cp.Problem(cp.Minimize(cp.sum_squares(self.g + lin)), cons)
        sv = np.linalg.svd(Ab, compute_uv=False)
        self.census = dict(A_zero=int((nA < delta).sum()), A_tie=int(((nA >= delta) & (sv[:, 0] - sv[:, 1] < delta)).sum()),
                           B_zero=int((nB < delta).sum()), C_zero=int((nC < delta).sum()),
                           nA=H * H, nB=H * nq, nC=nq * H)

    @staticmethod
    def flat(gA, gB, gC, H, nq):
        return np.concatenate([gA.reshape(H, 2, H, 2).transpose(0, 2, 1, 3).reshape(-1),
                               gB.reshape(H, 2, nq).transpose(0, 2, 1).reshape(-1),
                               gC.reshape(nq, H, 2).reshape(-1)])

    def residual(self, gA, gB, gC):
        self.g.value = self.flat(gA, gB, gC, self.H, self.nq)
        self.prob.solve(solver=cp.CLARABEL)
        assert self.prob.status in ("optimal", "optimal_inaccurate"), self.prob.status
        return float(np.sqrt(max(self.prob.value, 0.0))), self.prob.status


def run(seed):
    torch.set_num_threads(2)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = m2q.build4("p", seed)
    model.load_state_dict(torch.load(os.path.join(REGEN, f"t4_p_s{seed}.pt")))
    ref = json.load(open(os.path.join(HERE, "results_m2q", f"t4_p_s{seed}.json")))
    with torch.no_grad():
        train = float(((model(u_tr) - y_tr) ** 2).mean())
        test = float(((model(u_te) - y_te) ** 2).mean())
    v1 = dict(train=train, train_ref=ref["train"], test=test, test_ref=ref["test"],
              ok=abs(train - ref["train"]) <= 1e-5 * ref["train"] and abs(test - ref["test"]) <= 1e-5 * ref["test"])

    free = as_free(model)
    m = free.m
    H, k, nq = m.H, m.k, m.nq
    loss = ((free(u_tr) - y_tr) ** 2).mean()
    params = dict(free.named_parameters())
    names = [n for n, p in params.items() if p.requires_grad]
    grads = torch.autograd.grad(loss, [params[n] for n in names], allow_unused=True)
    G = {n: (g.detach() if g is not None else torch.zeros_like(params[n])) for n, g in zip(names, grads)}
    con = ("m.A_", "m.B_", "m.C")
    gA, gB, gC = (G[n].double().numpy() for n in con)
    g_con = float(np.sqrt((gA ** 2).sum() + (gB ** 2).sum() + (gC ** 2).sum()))
    g_unc = float(torch.sqrt(sum((G[n].double() ** 2).sum() for n in names if n not in con and n != "m.c")))

    with torch.no_grad():
        A, B, C = (x.detach().double().numpy() for x in (m.A_, m.B_, m.C))
        r = m2.rows(m, m.A_, m.B_)
    sat = [h for h in range(H) if float(r[h]) >= RHO - 1e-4]

    # v2 : NNLS avec les normales de la dérivation automatique (λ ≥ 0)
    X = [m.A_, m.B_, m.C]
    rr = m2q.mb.r_raw(H, k, nq, *X)
    Ncols = []
    for h in sat:
        gr = torch.autograd.grad(rr[h], X, retain_graph=True, allow_unused=True)
        Ncols.append(torch.cat([(a if a is not None else torch.zeros_like(x)).reshape(-1) for a, x in zip(gr, X)]).double())
    Nm = torch.stack(Ncols, 1).numpy()
    gv = np.concatenate([gA.reshape(-1), gB.reshape(-1), gC.reshape(-1)])
    _, r_nnls = nnls(Nm, -gv, maxiter=10000)
    coef = np.linalg.lstsq(Nm, gv, rcond=None)[0]
    r_signed = float(np.linalg.norm(gv - Nm @ coef))

    rng = np.random.default_rng(1000 + seed)
    rand = []
    for _ in range(N_RAND):
        z = rng.standard_normal(gv.shape)
        z *= g_con / np.linalg.norm(z)
        nA_, nB_ = gA.size, gB.size
        rand.append((z[:nA_].reshape(gA.shape), z[nA_:nA_ + nB_].reshape(gB.shape), z[nA_ + nB_:].reshape(gC.shape)))

    per_delta = []
    for d in DELTAS:
        t0 = time.time()
        P = KKT(A, B, C, sat, H, k, nq, d)
        R, st = P.residual(gA, gB, gC)
        Rr = [P.residual(*z)[0] / g_con for z in rand]
        per_delta.append(dict(delta=d, R=R, R_rel=R / g_con, status=st, rand_med=float(np.median(Rr)),
                              rand_min=float(np.min(Rr)), census=P.census, time=time.time() - t0))
        print(f"  s{seed} δ={d:g}: R/‖g‖={R / g_con:.3g} ({st}) | témoin méd. {np.median(Rr):.3g} "
              f"min {np.min(Rr):.3g} | plis {P.census} | {time.time() - t0:.0f}s", flush=True)
    v2 = dict(R0=per_delta[0]["R"], R_nnls=float(r_nnls), ok=abs(per_delta[0]["R"] - r_nnls) <= 1e-3 * g_con)
    return dict(seed=seed, v1=v1, v2=v2, g_con=g_con, g_unc=g_unc, unc_over_con=g_unc / g_con,
                r_signed_rel=r_signed / g_con, n_sat=len(sat), loss=float(loss), per_delta=per_delta)


def decide(res):
    # δ de décision : 1e-2, sauf si le témoin v3 échoue (règle du critère, § 6).
    def v3_ok(d):
        meds = [next(p for p in r["per_delta"] if p["delta"] == d)["rand_med"] for r in res]
        return sum(x >= 0.5 for x in meds) >= 3, meds
    dd = D_DEC
    ok, meds = v3_ok(dd)
    if not ok:
        cands = [d for d in DELTAS if v3_ok(d)[0]]
        dd = max(cands) if cands else None
    if dd is None:
        return dict(delta_dec=None, verdict="invalide (v3 échoue partout)")
    rel = [next(p for p in r["per_delta"] if p["delta"] == dd)["R_rel"] for r in res]
    conf, refu = sum(x <= 0.1 for x in rel), sum(x >= 0.5 for x in rel)
    verdict = "piste 2 confirmée" if conf >= 3 else "piste 2 réfutée" if refu >= 3 else "ambiguë"
    return dict(delta_dec=dd, R_rel=rel, v3_meds=v3_ok(dd)[1], n_conf=conf, n_refu=refu, verdict=verdict,
                v1_all=all(r["v1"]["ok"] for r in res), v2_all=all(r["v2"]["ok"] for r in res))


if __name__ == "__main__":
    check_nuc2()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    res = []
    for s in seeds:
        r = run(s)
        print(f"s{s}: v1 {r['v1']['ok']} (train {r['v1']['train']:.6g}/{r['v1']['train_ref']:.6g}, "
              f"test {r['v1']['test']:.6g}/{r['v1']['test_ref']:.6g}) | v2 {r['v2']['ok']} "
              f"(R0 {r['v2']['R0']:.4g}, NNLS {r['v2']['R_nnls']:.4g}) | ‖g_unc‖/‖g_con‖ {r['unc_over_con']:.3g} "
              f"| résidu signé (U1) {r['r_signed_rel']:.3g} | sat {r['n_sat']}", flush=True)
        res.append(r)
        json.dump(res, open(os.path.join(OUT, "kkt.json"), "w"), indent=1)
    if len(res) == 5:
        dec = decide(res)
        print("DÉCISION :", dec, flush=True)
        json.dump(dict(results=res, decision=dec), open(os.path.join(OUT, "kkt.json"), "w"), indent=1)
