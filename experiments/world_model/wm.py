"""
Expérience 2 — world model du Franka : modèles, entraînement, évaluation.

Tous les modèles prédisent ô_1..ô_H à partir de o_0 et des actions a_0..a_{H-1}
(rollout en boucle ouverte, comme dans l'imagination d'un agent model-based).

- mlp_1step : ô_{t+1} = ô_t + MLP(ô_t, a_t), appris pas à pas (teacher forcing, type MBPO)
- mlp_ms    : même modèle, appris sur des rollouts de H pas (BPTT)
- gru       : x_0 = enc(o_0), x_{t+1} = GRU(x_t, a_t), ô = dec(x)
- ren       : x_0 = enc(o_0), x_{t+1} = ExplicitREN(x_t, a_t) (contractant, garanti), ô = dec(x)

Usage : python wm.py --model ren --H 100 --seed 0 --data franka.npz
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
OUT = os.environ.get("OUT_DIR", "experiments/world_model/results")
NO, NA = 18, 8


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class MLPStep(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.f = mlp(NO + NA, h, NO)

    def forward(self, o0, a, teacher=None):
        o, outs = o0, []
        for t in range(a.shape[1]):
            inp = o if teacher is None else teacher[:, t]
            o = inp + 0.1 * self.f(torch.cat([inp, a[:, t]], -1))
            outs.append(o)
        return torch.stack(outs, 1)


class LatentWM(nn.Module):
    def __init__(self, cell, n, h=256):
        super().__init__()
        self.kind, self.n = cell, n
        self.enc = mlp(NO, h, n, depth=1)
        self.dec = mlp(n, h, NO, depth=1)
        if cell == "gru":
            self.cell = nn.GRUCell(NA, n)
        else:
            self.cell = ExplicitREN(NA, 1, n, n, gamma=None, alpha_bar=1.0, init_scale=1.0)

    def forward(self, o0, a, teacher=None):
        x = self.enc(o0)
        if self.kind == "gru":
            xs = []
            for t in range(a.shape[1]):
                x = self.cell(a[:, t], x)
                xs.append(x)
        else:
            fr = self.cell.frame()
            xs = []
            for t in range(a.shape[1]):
                x, _ = self.cell.step(fr, x, a[:, t])
                xs.append(x)
        return self.dec(torch.stack(xs, 1))


def build(model, n):
    if model.startswith("mlp"):
        return MLPStep()
    return LatentWM(model, n)


# ----------------------------------------------------------------------------------------
class Data:
    def __init__(self, path):
        z = np.load(path)
        self.mo, self.so = z["Otr"].reshape(-1, NO).mean(0), z["Otr"].reshape(-1, NO).std(0) + 1e-3
        self.ma, self.sa = z["Atr"].reshape(-1, NA).mean(0), z["Atr"].reshape(-1, NA).std(0) + 1e-3
        n = lambda O, A: (torch.tensor((O - self.mo) / self.so), torch.tensor((A - self.ma) / self.sa))  # noqa
        self.tr, self.va, self.te = n(z["Otr"], z["Atr"]), n(z["Ova"], z["Ava"]), n(z["Ote"], z["Ate"])

    def batch(self, B, H, gen):
        O, A = self.tr
        i = torch.randint(0, O.shape[0], (B,), generator=gen)
        s = torch.randint(0, O.shape[1] - H, (B,), generator=gen)
        idx = s[:, None] + torch.arange(H + 1)
        o = O[i[:, None], idx]
        a = A[i[:, None], idx[:, :-1]]
        return o[:, 0], a, o[:, 1:]


def train(model, data, H, iters, seed, lr=1e-3):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    one_step = isinstance(model, MLPStep) and model.one_step
    log = dict(loss=[], grad=[], nonfinite=0)
    t0 = time.time()
    for it in range(iters):
        o0, a, o = data.batch(64, H, gen)
        if one_step:
            pred = model(o0, a, teacher=torch.cat([o0[:, None], o[:, :-1]], 1))
        else:
            pred = model(o0, a)
        loss = ((pred - o) ** 2).mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            log["nonfinite"] += 1
            sched.step()
            continue
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(g):
            log["nonfinite"] += 1
            opt.zero_grad()
        else:
            opt.step()
        sched.step()
        if it % 20 == 0:
            log["loss"].append(loss.item())
            log["grad"].append(g.item())
    log["time"] = time.time() - t0
    return log


@torch.no_grad()
def evaluate(model, data, horizons=(10, 50, 100, 250, 500, 1000, 1500)):
    O, A = data.te
    pred = model(O[:, 0], A)                      # 1500 pas en boucle ouverte
    err = ((pred - O[:, 1:]) ** 2).mean(-1)       # (N, T), échelle normalisée
    finite = torch.isfinite(pred).all(-1)
    res = {}
    for h in horizons:
        e = err[:, :h]
        ok = finite[:, :h].all(1)
        res[f"nmse@{h}"] = float(e[ok].mean()) if ok.any() else math.inf
        res[f"diverged@{h}"] = float(1 - ((ok) & (pred[:, :h].abs().amax((1, 2)) < 1e3)).float().mean())
    res["max_abs_pred"] = float(torch.nan_to_num(pred.abs(), nan=1e30).max())
    # erreur sur la seconde moitié (régime établi, après oubli de o_0)
    res["nmse_last_half"] = float(err[:, 750:][finite[:, 750:].all(1)].mean()) if finite.all() else math.inf
    vo, va = data.va
    vp = model(vo[:, 0], va)
    res["val_nmse_250"] = float(((vp - vo[:, 1:]) ** 2).mean())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ren")
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="franka.npz")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    data = Data(a.data)
    model = build(a.model, a.n)
    if isinstance(model, MLPStep):
        model.one_step = a.model == "mlp_1step"
    nparams = sum(p.numel() for p in model.parameters())
    log = train(model, data, a.H, a.iters, a.seed)
    ev = evaluate(model, data)
    name = f"{a.model}_H{a.H}_n{a.n}_s{a.seed}"
    json.dump(dict(args=vars(a), nparams=nparams, log=log, eval=ev), open(f"{OUT}/{name}.json", "w"))
    torch.save(model.state_dict(), f"{OUT}/{name}.pt")
    print(name, nparams, f"{log['time']:.0f}s", {k: round(v, 4) for k, v in ev.items()}, flush=True)
