"""
Q0 — M2-r : le certificat est-il trop grossier ? Critères : `analysis/m2r.md` (poussés avant calcul, 120a87d).

  t8    : finesse du certificat sur les modèles finaux de T4 (libre, fix2, P) : r, r_mid, R_lin, r_box_lo, R_loc ;
  train : bras mid (T9), alpha (T10), midalpha (T11) : Adam lr 1e-2, 3000 itér., init W₀' ;
  dval  : non-expansion empirique des modèles finaux, dans la norme de chaque bras.
Le type par défaut reste float32 (le tirage aléatoire du banc en dépend) ; le float64 est obtenu par .double().

Usage :
  PYTHONPATH=. python experiments/q0/m2r.py t8
  PYTHONPATH=. python experiments/q0/m2r.py train --arms mid,alpha,midalpha --seeds 0,1,2,3,4
  PYTHONPATH=. python experiments/q0/m2r.py dval
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
import m2bis as mb  # noqa: E402
import m2q  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2r"))
D4 = os.path.join(HERE, "results_m2q")
RHO, TOL = m2.RHO, m2.TOL
ARMS = ("mid", "alpha", "midalpha")


# ------------------------------------------------------------------ certificats
def bnorm(M, H, k):
    """Normes spectrales des blocs k×k : M (..., H·k, H·k) → (..., H, H)."""
    lead = M.shape[:-2]
    return torch.linalg.matrix_norm(M.reshape(*lead, H, k, H, k).transpose(-3, -2), ord=2)


def cert(A, B, C, H, k, nq, kind="r", alpha=None):
    """Certificat par tête dans la norme maxₕ α_h‖x_h‖ : r (triangulaire) ou r_mid (centre-rayon)."""
    al = torch.ones(H, dtype=A.dtype) if alpha is None else alpha
    nB = B.reshape(H, k, nq).norm(dim=1)                                 # (H, nq)
    cq = (C.reshape(nq, H, k).norm(dim=2) / al[None, :]).sum(1)          # (nq,)
    if kind == "r":
        nM, w = bnorm(A, H, k), 1.0
    else:
        nM, w = bnorm(A + 0.5 * B @ C, H, k), 0.5
    return (nM * al[:, None] / al[None, :]).sum(1) + w * al * (nB @ cq)


class CertBlock(m2.BlockGatedRNN):
    """Φ « proj » avec le certificat du bras : s_h = min(1, ρ/c_h). Dans S, s = 1 exactement."""

    def cert_rows(self):
        al = torch.exp(self.log_alpha) if self.learn_alpha else None
        return cert(self.A_, self.B_, self.C, self.H, self.k, self.nq, self.kind, al)

    def weights(self) -> dict:
        s = torch.clamp(self.rho / self.cert_rows(), max=1.0).repeat_interleave(self.k)[:, None]
        return dict(A=self.A_ * s, B=self.B_ * s)


def build_r(arm, seed):
    model = mb.build("pint", seed)                                         # W₀' (0,9ρ selon r)
    m = model.m
    m.__class__ = CertBlock
    m.kind = "mid" if arm in ("mid", "midalpha") else "r"
    m.learn_alpha = arm in ("alpha", "midalpha")
    m.log_alpha = torch.nn.Parameter(torch.zeros(m.H), requires_grad=m.learn_alpha)
    return model


def alpha_of(m):
    return torch.exp(m.log_alpha.detach()) if getattr(m, "learn_alpha", False) else None


@torch.no_grad()
def cert_W(m, kind=None, alpha="arm"):
    W = m.weights()
    al = alpha_of(m) if alpha == "arm" else alpha
    return cert(W["A"], W["B"], m.C, m.H, m.k, m.nq, kind or m.kind, al)


# ------------------------------------------------------------------ entraînement (T9–T11)
def train(arm, seed, iters=3000, lr=1e-2):
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = build_r(arm, seed)
    m = model.m
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses, log, status, t0 = [], [], "ok", time.time()
    for it in range(iters + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        losses.append(float(loss.detach()))
        if it % 50 == 0:
            al = alpha_of(m)
            log.append(dict(it=it, loss=losses[-1], c=float(cert_W(m).max()),
                            r_fixe=float(cert_W(m, "r", None).max()),
                            amin=float(al.min()) if al is not None else 1.0,
                            amax=float(al.max()) if al is not None else 1.0))
        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"{arm}_s{seed}.pt"))
    al = alpha_of(m)
    return dict(arm=arm, seed=seed, lr=lr, iters=iters, status=status, test=test, train=losses[-1],
                c_end=float(cert_W(m).max()), r_fixe_end=float(cert_W(m, "r", None).max()),
                alpha=[float(x) for x in al] if al is not None else None,
                time=time.time() - t0, log=log, losses=losses)


def _job(a):
    r = train(*a)
    json.dump(r, open(os.path.join(OUT, f"{r['arm']}_s{r['seed']}.json"), "w"))
    return (f"train {r['arm']:8s} s{r['seed']}: {r['status']} train={r['train']:.4f} test={r['test']:.4f} "
            f"c_end={r['c_end']:.4f} r_fixe_end={r['r_fixe_end']:.3f} {r['time']:.0f}s")


# ------------------------------------------------------------------ T8
def load4(arm, seed):
    model = m2q.build4(arm, seed)
    model.load_state_dict(torch.load(os.path.join(D4, f"t4_{arm}_s{seed}.pt")))
    model = model.double()
    if arm == "fix2":
        model.m.rho = 2.0
    return model


@torch.no_grad()
def rows_at(A, B, C, H, k, d):
    """maxₕ Σₖ ‖[A + B diag(d) C]_hk‖ pour un lot de vecteurs de pentes d (N, nq)."""
    M = A[None] + (B[None] * d[:, None, :]) @ C[None]
    return bnorm(M, H, k).sum(-1).max(-1).values


@torch.no_grad()
def t8_one(arm, seed):
    torch.set_num_threads(2)
    model = load4(arm, seed)
    m = model.m
    W = m.weights()
    A, B, C, H, k, nq = W["A"], W["B"], m.C, m.H, m.k, m.nq
    r = float(cert(A, B, C, H, k, nq, "r").max())
    rmid = float(cert(A, B, C, H, k, nq, "mid").max())
    one = torch.ones(1, nq, dtype=A.dtype)
    R_lin = float(rows_at(A, B, C, H, k, one)[0])
    g = torch.Generator().manual_seed(123 + seed)
    ds = torch.cat([(torch.rand(800, nq, generator=g) < p).double() for p in (0.1, 0.3, 0.5, 0.7, 0.9)])
    vals = rows_at(A, B, C, H, k, ds)
    best_v = float(vals.max())
    for i in torch.argsort(vals, descending=True)[:10]:
        d, v = ds[i].clone(), float(vals[i])
        while True:
            flips = d[None].repeat(nq, 1)
            flips[torch.arange(nq), torch.arange(nq)] = 1 - flips[torch.arange(nq), torch.arange(nq)]
            fv = rows_at(A, B, C, H, k, flips)
            j = int(fv.argmax())
            if float(fv[j]) <= v + 1e-12:
                break
            d, v = flips[j], float(fv[j])
        best_v = max(best_v, v)
    _, _, u_te, _ = m2.data(seed)
    u = u_te[:32].double()
    xs = m.forward(u)                                                    # (N, T, nx)
    xp = torch.cat([torch.zeros_like(xs[:, :1]), xs[:, :-1]], 1)
    a = xp @ C.T + u @ m.D.T + m.b
    sl = (1 - torch.tanh(a) ** 2).reshape(-1, nq)
    R_loc = float(torch.cat([rows_at(A, B, C, H, k, sl[i:i + 2000]) for i in range(0, len(sl), 2000)]).max())
    return dict(arm=arm, seed=seed, r=r, r_mid=rmid, R_lin=R_lin, vertex=best_v, R_loc=R_loc,
                r_box_lo=max(best_v, R_loc))


# ------------------------------------------------------------------ d-val
@torch.no_grad()
def E_test(model, alpha, seed, n=32, T=2000, pairs=4):
    m = model.m
    u = m2.osc.inputs(n, T, 30 + seed).double().repeat(pairs, 1, 1)
    al = (torch.ones(m.H, dtype=torch.float64) if alpha is None else alpha.double())
    g = torch.Generator().manual_seed(77 + seed)
    E = 0.0
    for scale in ("grande", "petite"):
        x0 = torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64)
        x1 = (torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64) if scale == "grande"
              else x0 + 1e-3 * torch.randn(n * pairs, m.nx, generator=g, dtype=torch.float64))

        def dist(a, b):
            return ((a - b).view(*a.shape[:-1], m.H, m.k).norm(dim=-1) * al).max(-1).values
        d0 = dist(x0, x1)
        E = max(E, float((dist(m.forward(u, x0), m.forward(u, x1)) / d0[:, None]).max()))
    return E


def dval(arm, seed):
    torch.set_num_threads(2)
    model = build_r(arm, seed)
    model.load_state_dict(torch.load(os.path.join(OUT, f"{arm}_s{seed}.pt")))
    model = model.double()
    return dict(arm=arm, seed=seed, E=E_test(model, alpha_of(model.m), seed),
                c_end=float(cert_W(model.m).max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("t8", "train", "dval"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "t8":
        res = []
        for arm in ("p", "free", "fix2"):
            for s in seeds:
                r = t8_one(arm, s)
                res.append(r)
                print(f"t8 {arm:5s} s{s}: r={r['r']:.4g} r_mid={r['r_mid']:.4g} R_lin={r['R_lin']:.4g} "
                      f"sommet={r['vertex']:.4g} R_loc={r['R_loc']:.4g} → r_box_lo={r['r_box_lo']:.4g}", flush=True)
        json.dump(res, open(os.path.join(OUT, "t8.json"), "w"), indent=1)
    elif a.what == "train":
        jobs = [(ar, s, a.iters) for s, ar in itertools.product(seeds, a.arms.split(","))]
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
            for line in pool.imap_unordered(_job, jobs):
                print(line, flush=True)
    else:
        res = [dval(ar, s) for ar in a.arms.split(",") for s in seeds]
        json.dump(res, open(os.path.join(OUT, "dval.json"), "w"), indent=1)
        for r in res:
            print(f"dval {r['arm']:8s} s{r['seed']}: E={r['E']:.6f} c_end={r['c_end']:.5f}", flush=True)
