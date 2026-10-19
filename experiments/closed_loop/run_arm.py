"""Entraînement et évaluation de l'expérience 1.

Usage :
    python run_arm.py random   [--draws 150]
    python run_arm.py train    --kind pb_ren --seed 0 [--gamma 200] [--iters 600]
Les résultats vont dans $OUT_DIR (défaut : experiments/closed_loop/results).
"""
import argparse
import json
import os
import time

import torch

from arm_boosting import (NOMINAL, MLPPolicy, PBREN, PBRNN, cost, evaluate, rollout, sample_x0)

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
OUT = os.environ.get("OUT_DIR", "experiments/closed_loop/results")
T_TRAIN = 150          # 3 s
T_LONG = 1500          # 30 s (10 × l'horizon d'entraînement)


def make(kind, scale=1.0, gamma=200.0):
    return {"mlp": lambda: MLPPolicy(scale=scale),
            "pb_rnn": lambda: PBRNN(scale=scale),
            "pb_ren": lambda: PBREN(gamma=gamma, scale=scale)}[kind]()


def random_parameters(n_draws=150):
    """Boucle fermée avec des paramètres tirés au hasard (aucun entraînement)."""
    res = {}
    x0 = sample_x0(16, gen=torch.Generator().manual_seed(1))
    for kind in ("mlp", "pb_rnn", "pb_ren"):
        for scale in (1.0, 3.0):
            stats = []
            for d in range(n_draws):
                torch.manual_seed(1000 + d)
                stats.append(evaluate(make(kind, scale), x0, T_LONG))
            agg = {k: sum(s[k] for s in stats) / n_draws for k in ("reached", "diverged")}
            agg["all_reached"] = sum(s["reached"] == 1.0 for s in stats) / n_draws
            agg["n_draws"] = n_draws
            res[f"{kind}/scale={scale}"] = agg
            print(kind, scale, agg, flush=True)
    json.dump(res, open(f"{OUT}/random_parameters.json", "w"), indent=1)


def train(kind, seed, gamma=200.0, iters=600, lr=1e-2):
    torch.manual_seed(seed)
    pol = make(kind, gamma=gamma)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    gen = torch.Generator().manual_seed(seed)
    hist, nonfinite = [], 0
    t0 = time.time()
    for it in range(iters):
        xs, us = rollout(pol, sample_x0(32, gen=gen), T_TRAIN)
        loss = cost(xs, us)
        opt.zero_grad()
        if not torch.isfinite(loss):
            nonfinite += 1
            hist.append(float("nan"))
            sched.step()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 10.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
    return pol, hist, nonfinite, time.time() - t0


def suites(pol, seed=0):
    g = torch.Generator().manual_seed(10_000 + seed)
    x0 = sample_x0(200, gen=g)
    x0w = sample_x0(200, spread=0.5, gen=g)
    pushes = torch.zeros(200, T_LONG, 4)
    tk = torch.randint(25, 500, (200,), generator=g)
    pushes[torch.arange(200), tk, 2:] = 0.04 * torch.randn(200, 2, generator=g) / 0.02  # ≈ ±2 rad/s
    out = {"train_3s": evaluate(pol, x0, T_TRAIN),
           "long_30s": evaluate(pol, x0, T_LONG),
           "wide_ic_30s": evaluate(pol, x0w, T_LONG),
           "pushes_30s": evaluate(pol, x0, T_LONG, pushes=pushes)}
    for k in (1.5, 2.0, 3.0):
        out[f"payload_x{k}_30s"] = evaluate(pol, x0, T_LONG, plant=dict(NOMINAL, m2=NOMINAL["m2"] * k))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["random", "train"])
    ap.add_argument("--kind", default="pb_ren")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=200.0)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--draws", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.phase == "random":
        random_parameters(a.draws)
    else:
        pol, hist, nonfinite, dt = train(a.kind, a.seed, a.gamma, a.iters)
        ev = suites(pol, a.seed)
        name = f"{a.kind}_g{a.gamma:g}_s{a.seed}"
        json.dump(dict(kind=a.kind, seed=a.seed, gamma=a.gamma, iters=a.iters, hist=hist,
                       nonfinite=nonfinite, train_time=dt, eval=ev), open(f"{OUT}/{name}.json", "w"))
        torch.save(pol.state_dict(), f"{OUT}/{name}.pt")
        print(name, f"{dt:.0f}s loss={hist[-1]:.3f}",
              {k: (round(v["success"], 2), round(v["diverged"], 2)) for k, v in ev.items()}, flush=True)
