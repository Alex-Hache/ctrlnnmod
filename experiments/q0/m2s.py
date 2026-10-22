"""
Q0 — M2-s : le gradient choisit-il le mauvais type de solution ? Critères : `analysis/m2s.md` (b780980).
Bras : lin, linfree, gate0, lingate0, basin2, basin1. Adam lr 1e-2, 3000 itér. Type par défaut float32 (banc).
Usage :
  PYTHONPATH=. python experiments/q0/m2s.py check
  PYTHONPATH=. python experiments/q0/m2s.py train --seeds 0,1,2,3,4
  PYTHONPATH=. python experiments/q0/m2s.py dval
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
import m2q  # noqa: E402
import m2r  # noqa: E402

OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "results_m2s"))
RHO = m2.RHO
ARMS = ("lin", "linfree", "gate0", "lingate0", "basin2", "basin1")
CONSTRAINED = ("lin", "gate0", "lingate0", "basin2", "basin1")


@torch.no_grad()
def teacher_on_boundary(model, seed):
    """Représentation exacte du maître (graine) avec les 4 têtes utiles sur ∂S ; B̃ = C = D = 0."""
    m = model.m
    A_t, B_t, C_t = m2.osc.teacher(seed)
    sc = float(m2.osc.sim(A_t, B_t, C_t, m2.osc.inputs(256, 100, 10 + seed)).std())
    for p in model.parameters():
        p.zero_()
    A, E, gb = torch.zeros(16, 16), torch.zeros(16, 2), torch.full((8,), -3.0)
    for h, th in enumerate(m2.osc.TH):
        lam = 0.99 * complex(math.cos(math.radians(th)), math.sin(math.radians(th)))
        f = lambda z: abs(lam - (1 - z)) - RHO * z  # noqa: E731
        lo, hi = 1e-3, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        z = (lo + hi) / 2
        A[2 * h:2 * h + 2, 2 * h:2 * h + 2] = (A_t[2 * h:2 * h + 2, 2 * h:2 * h + 2] - (1 - z) * torch.eye(2)) / z
        E[2 * h:2 * h + 2] = B_t[2 * h:2 * h + 2] / z
        q = (z - m.z_min) / (1 - m.z_min)
        gb[h] = math.log(q / (1 - q))
    for h in range(4, 8):
        A[2 * h:2 * h + 2, 2 * h:2 * h + 2] = RHO * torch.eye(2)
    m.A_.copy_(A)
    m.E.copy_(E)
    m.gate_proj.bias.copy_(gb)
    model.C.weight[:, :8] = C_t / sc


def build_s(arm, seed):
    model = mb.build("pint", seed)                         # W₀' (0,9ρ)
    m = model.m
    m.__class__ = m2.FreeBlockGatedRNN if arm == "linfree" else m2q.ExactProjBlock
    with torch.no_grad():
        if arm in ("lin", "linfree", "lingate0"):
            m.B_.zero_()
            m.B_.requires_grad_(False)
        if arm in ("gate0", "lingate0"):
            m.gate_proj.bias.zero_()
        if arm.startswith("basin"):
            teacher_on_boundary(model, seed)
            sig = {"basin2": 1e-2, "basin1": 1e-1}[arm]
            g = torch.Generator().manual_seed(900 + seed)
            for p in model.parameters():
                p.mul_(1 + sig * torch.randn(p.shape, generator=g))
    return model


def gate_z(m):
    with torch.no_grad():
        return m.z_min + (1 - m.z_min) * torch.sigmoid(m.gate_proj.bias)


def run(arm, seed, iters=3000, lr=1e-2):
    torch.set_num_threads(1)
    u_tr, y_tr, u_te, y_te = m2.data(seed)
    model = build_s(arm, seed)
    m = model.m
    with torch.no_grad():
        test0 = float(((model(u_te) - y_te) ** 2).mean())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses, log, status, t0 = [], [], "ok", time.time()
    for it in range(iters + 1):
        loss = ((model(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        losses.append(float(loss.detach()))
        if it % 50 == 0:
            log.append(dict(it=it, loss=losses[-1], rW=float(m2q.r_W(m).max()),
                            z=[round(float(x), 3) for x in gate_z(m)]))
        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        opt.step()
    with torch.no_grad():
        test = float(((model(u_te) - y_te) ** 2).mean()) if status == "ok" else float("inf")
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"{arm}_s{seed}.pt"))
    return dict(arm=arm, seed=seed, status=status, test0=test0, test=test, train=losses[-1],
                rW_end=float(m2q.r_W(m).max()), B_norm_end=float(m.B_.detach().norm()),
                z_end=[float(x) for x in gate_z(m)], time=time.time() - t0, log=log, losses=losses)


def _job(a):
    r = run(*a)
    json.dump(r, open(os.path.join(OUT, f"{r['arm']}_s{r['seed']}.json"), "w"))
    return (f"train {r['arm']:8s} s{r['seed']}: {r['status']} test0={r['test0']:.4f} train={r['train']:.4f} "
            f"test={r['test']:.4f} rW_end={r['rW_end']:.3f} |B|={r['B_norm_end']:.2g} {r['time']:.0f}s")


def dval(arm, seed):
    torch.set_num_threads(2)
    model = build_s(arm, seed)
    model.load_state_dict(torch.load(os.path.join(OUT, f"{arm}_s{seed}.pt")))
    model = model.double()
    return dict(arm=arm, seed=seed, E=m2r.E_test(model, None, seed))


@torch.no_grad()
def check():
    """Avant calcul : la représentation du maître donne une erreur ~0 et r = ρ sur les têtes utiles."""
    for seed in range(5):
        u_tr, y_tr, u_te, y_te = m2.data(seed)
        model = mb.build("pint", seed)
        model.m.__class__ = m2q.ExactProjBlock
        teacher_on_boundary(model, seed)
        r = m2q.r_W(model.m)
        print(f"check s{seed}: test {float(((model(u_te) - y_te) ** 2).mean()):.2e} train "
              f"{float(((model(u_tr) - y_tr) ** 2).mean()):.2e} | r par tête " + " ".join(f"{float(x):.4f}" for x in r)
              + " | z " + " ".join(f"{float(x):.3f}" for x in gate_z(model.m)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "train", "dval"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "check":
        check()
    elif a.what == "train":
        jobs = [(ar, s, a.iters) for s, ar in itertools.product(seeds, a.arms.split(","))]
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
            for line in pool.imap_unordered(_job, jobs):
                print(line, flush=True)
    else:
        res = [dval(ar, s) for ar in a.arms.split(",") if ar in CONSTRAINED for s in seeds]
        json.dump(res, open(os.path.join(OUT, "dval.json"), "w"), indent=1)
        for r in res:
            print(f"dval {r['arm']:8s} s{r['seed']}: E={r['E']:.6f}", flush=True)
