"""
Session « gradient non nul, perte plate » — test de la piste 3 (pas trop grand).
Critère : `experiments/q0/sessions/theorie_gradient_plat.md` § 6b (poussé avant calcul, df54e84).

Départ : P final régénéré (T4). Adam neuf, clip 10, perte pleine, 1000 pas, lr ∈ {1e-2 (témoin), 1e-3}.

Usage :
  REGEN=<dossier des .pt régénérés> PYTHONPATH=. python experiments/q0/gp_lr.py
"""
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
import m2q  # noqa: E402

REGEN = os.environ["REGEN"]
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_gp"))
ITERS = 1000


def run(args):
    lr, seed = args
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = m2q.build4("p", seed)
    model.load_state_dict(torch.load(os.path.join(REGEN, f"t4_p_s{seed}.pt")))
    m = model.m
    H, k, nq = m.H, m.k, m.nq
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, rraw, t0 = [], [], time.time()
    for it in range(ITERS + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            rraw.append(float(m2q.mb.r_raw(H, k, nq, m.A_, m.B_, m.C).max()))
        if it == ITERS:
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean())
        rW = float(m2q.r_W(m).max())
    L = losses
    b = (L[0] - L[-1]) / L[0]
    u = sum(1 for i in range(501, ITERS + 1) if L[i] > L[i - 1]) / 500
    return dict(lr=lr, seed=seed, L0=L[0], L_end=L[-1], b=b, u=u, test=test, rW_end=rW,
                rraw0=rraw[0], rraw_end=rraw[-1], losses=L[::10], time=time.time() - t0)


def decide(res):
    def arm(lr):
        return sorted((r for r in res if r["lr"] == lr), key=lambda r: r["seed"])
    ctl, low = arm(1e-2), arm(1e-3)
    valid = sum(r["b"] < 0.10 and r["u"] >= 0.30 for r in ctl) >= 3
    conf = sum(r["b"] >= 0.20 and r["u"] < 0.20 for r in low)
    refu = sum(r["b"] < 0.05 for r in low)
    verdict = ("invalide (témoin)" if not valid else "piste 3 confirmée" if conf >= 3
               else "piste 3 réfutée" if refu >= 3 else "partielle")
    return dict(valid=valid, n_conf=conf, n_refu=refu, verdict=verdict,
                ctl=[(r["b"], r["u"]) for r in ctl], low=[(r["b"], r["u"]) for r in low])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = list(itertools.product((1e-2, 1e-3), range(5)))
    res = []
    with mp.get_context("fork").Pool(2) as pool:
        for r in pool.imap_unordered(run, jobs):
            print(f"lr {r['lr']:g} s{r['seed']}: L {r['L0']:.4g} → {r['L_end']:.4g} (b = {r['b']:.3f}) | u = {r['u']:.2f} "
                  f"| test {r['test']:.3g} | r brut {r['rraw0']:.2f} → {r['rraw_end']:.2f} | rW {r['rW_end']:.4f} "
                  f"| {r['time']:.0f}s", flush=True)
            res.append(r)
    dec = decide(res)
    print("DÉCISION :", dec, flush=True)
    json.dump(dict(results=res, decision=dec), open(os.path.join(OUT, "lr.json"), "w"), indent=1)
