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
from ctrlnmod.models.discrete.selective_ren import SelectiveREN

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
        elif cell == "sren":
            self.cell = SelectiveREN(NA, n_heads=8, nx=n // 8, nq=2 * (n // 8), alpha_bar=0.9, g_min=0.02)
        else:
            self.cell = ExplicitREN(NA, 1, n, n, gamma=None, alpha_bar=1.0, init_scale=1.0)

    # interface générique (utilisée par imagine.py)
    def prepare(self):
        return self.cell.frame() if self.kind in ("ren", "sren") else None

    def init_state(self, o0):
        return self.enc(o0)

    def advance(self, ctx, x, a):
        if self.kind == "gru":
            x = self.cell(a, x)
        elif self.kind == "sren":
            x = self.cell.step(ctx, x, a)
        else:
            x = self.cell.step(ctx, x, a)[0]
        return x, self.dec(x)

    def forward(self, o0, a, teacher=None):
        x = self.enc(o0)
        if self.kind == "sren":
            fr = self.cell.frame()
            xs = []
            for t in range(a.shape[1]):
                x = self.cell.step(fr, x, a[:, t])
                xs.append(x)
            return self.dec(torch.stack(xs, 1))
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


class Mamba2Block(nn.Module):
    """Bloc de type Mamba-2 en PyTorch pur (récurrence pas à pas, sans noyau CUDA) :
    projection d'entrée, convolution causale courte, SSM sélectif à A scalaire par tête
    (Δ, B, C dépendants de l'entrée), porte silu(z), normalisation RMS, projection de sortie."""

    def __init__(self, d, expand=2, heads=8, N=16, k=4):
        super().__init__()
        self.d, self.di, self.h, self.N, self.k = d, expand * d, heads, N, k
        self.p = self.di // heads
        self.in_proj = nn.Linear(d, 2 * self.di + 2 * N + heads)
        self.conv_w = nn.Parameter(torch.randn(self.di + 2 * N, k) / k ** 0.5)
        self.conv_b = nn.Parameter(torch.zeros(self.di + 2 * N))
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, heads)))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.linspace(1e-3, 1e-1, heads))))
        self.Dskip = nn.Parameter(torch.ones(heads))
        self.norm_w = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d)
        self.pre_norm = nn.Parameter(torch.ones(d))

    def init_state(self, B):
        return (torch.zeros(B, self.h, self.p, self.N), torch.zeros(B, self.k - 1, self.di + 2 * self.N))

    @staticmethod
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def step(self, state, u):
        h, buf = state
        B = u.shape[0]
        zxbcdt = self.in_proj(self.rms(u, self.pre_norm))
        z, xBC, dt = torch.split(zxbcdt, [self.di, self.di + 2 * self.N, self.h], -1)
        win = torch.cat([buf, xBC[:, None]], 1)                         # (B, k, C)
        xBC = torch.nn.functional.silu((win * self.conv_w.T[None]).sum(1) + self.conv_b)
        buf = win[:, 1:]
        x, Bm, Cm = torch.split(xBC, [self.di, self.N, self.N], -1)
        dt = torch.nn.functional.softplus(dt + self.dt_bias)            # (B, H)
        dA = torch.exp(-dt * torch.exp(self.A_log))                     # (B, H)
        xh = x.view(B, self.h, self.p)
        h = dA[..., None, None] * h + (dt[..., None, None] * xh[..., None]) * Bm[:, None, None, :]
        y = (h * Cm[:, None, None, :]).sum(-1) + self.Dskip[None, :, None] * xh
        y = self.rms(y.reshape(B, self.di) * torch.nn.functional.silu(z), self.norm_w)
        return (h, buf), u + self.out_proj(y)


class MambaWM(nn.Module):
    """World model à 2 blocs Mamba-2 : jeton 0 = enc(o_0), puis un jeton par action."""

    def __init__(self, d=96, layers=2, h=256):
        super().__init__()
        self.kind = "mamba2"
        self.enc = mlp(NO, h, d, depth=1)
        self.act_emb = nn.Linear(NA, d)
        self.blocks = nn.ModuleList([Mamba2Block(d) for _ in range(layers)])
        self.dec = mlp(d, h, NO, depth=1)

    def prepare(self):
        return None

    def _run(self, states, tok):
        new = []
        for blk, st in zip(self.blocks, states):
            st, tok = blk.step(st, tok)
            new.append(st)
        return new, tok

    def init_state(self, o0):
        states = [b.init_state(o0.shape[0]) for b in self.blocks]
        states, _ = self._run(states, self.enc(o0))
        return states

    def advance(self, ctx, states, a):
        states, feat = self._run(states, self.act_emb(a))
        return states, self.dec(feat)

    def forward(self, o0, a, teacher=None):
        st = self.init_state(o0)
        outs = []
        for t in range(a.shape[1]):
            st, o = self.advance(None, st, a[:, t])
            outs.append(o)
        return torch.stack(outs, 1)


def build(model, n):
    if model.startswith("mlp"):
        return MLPStep()
    if model == "mamba2":
        return MambaWM()
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
