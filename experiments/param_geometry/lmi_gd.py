"""Descente de gradient pure sur les modèles H∞ de ctrlnmod : ½‖(A,B,C) − (A*,B*,C*)‖² vers un maître
de la même classe. Itérations jusqu'à une erreur relative 1e-6 (meilleur pas). P aléatoire contre P = I."""
import sys
from multiprocessing import Pool
import numpy as np
import torch
sys.argv = ["x"]
exec(open("experiments/param_geometry/lmi_fingerprint.py").read().split("res = []")[0])
NX, ITERS, TOL = 4, 4000, 1e-6
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def model(kind, pd, init, seed):
    torch.manual_seed(seed)
    m = build(kind, NX, 2, 2, pd)
    if init == "I":
        with torch.no_grad():
            if pd == "geotorch":
                m.P = torch.eye(NX)
            else:
                m.parametrizations.P.original.copy_(torch.eye(NX))
    return m


def run(a):
    try:
        return _run(a)
    except Exception:                                  # noqa: BLE001 (divergence : sqrtm, eigh)
        kind, pd, init, seed, lr = a
        return (kind, pd, init, seed, lr, None, float("inf"), float("nan"))


def _run(a):
    kind, pd, init, seed, lr = a
    torch.set_num_threads(1)
    tgt = torch.cat([t.reshape(-1) for t in model(kind, "geotorch", "aléatoire", 100 + seed)._frame()]).detach()
    m = model(kind, pd, init, seed)
    kap = fingerprint(m)["kappa_range"]
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    n0 = float(tgt.norm() ** 2)
    for it in range(ITERS):
        out = torch.cat([t.reshape(-1) for t in m._frame()])
        loss = 0.5 * ((out - tgt) ** 2).sum()
        rel = 2 * float(loss) / n0
        if not np.isfinite(rel):
            return (kind, pd, init, seed, lr, None, float("inf"), kap)
        if rel < TOL:
            return (kind, pd, init, seed, lr, it, rel, kap)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (kind, pd, init, seed, lr, None, rel, kap)


if __name__ == "__main__":
    jobs = [(k, p, i, s, lr) for k in ("hinf_sqrtm", "hinf_riccati") for p in ("geotorch", "bm")
            for i in ("aléatoire", "I") for s in range(3) for lr in LRS]
    with Pool(8) as pool:
        R = pool.map(run, jobs)
    rows = ["| Paramétrisation | PD | P au départ | κ médian au départ | itérations (médiane, meilleur pas) | erreur finale sinon |",
            "|---|---|---|---|---|---|"]
    for k in ("hinf_sqrtm", "hinf_riccati"):
        for p in ("geotorch", "bm"):
            for i in ("aléatoire", "I"):
                its, fins, kaps = [], [], []
                for s in range(3):
                    rs = [r for r in R if r[:4] == (k, p, i, s)]
                    kaps.append(rs[0][7])
                    conv = [r[5] for r in rs if r[5] is not None]
                    its.append(min(conv) if conv else None)
                    fins.append(min(r[6] for r in rs))
                ok = [x for x in its if x is not None]
                itc = f"{int(np.median(ok))} ({len(ok)}/3 convergent)" if ok else "> 4000"
                rows.append(f"| {k} | {p} | {i} | {np.median(kaps):.1e} | {itc} | {max(fins):.1e} |")
    txt = "\n".join(rows)
    print(txt)
    open("experiments/param_geometry/analysis/lmi_gd.md", "w").write(txt + "\n")
