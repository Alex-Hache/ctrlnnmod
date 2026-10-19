"""
M3a — world model à partir d'images (Franka, MuJoCo), sur GPU.

Entrée : deux images 64×64 RGB (t, t+1) → encodeur CNN → état latent x_0.
Dynamique latente pilotée par les actions : GRU (cuDNN), REN explicite, REN sélectif (noyau
Triton), Mamba-2. Sortie : l'état articulaire futur (18 grandeurs, normalisées) sur H pas.
L'encodeur doit retrouver l'état (position et vitesse) à partir des pixels.

Usage : python pwm.py --model sren --n 1024 --H 50 --seed 0 --data /data/franka_pix.npz
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
from ctrlnmod.models.discrete.selective_ren import SelectiveREN, rollout

OUT = os.environ.get("OUT_DIR", "experiments/pixel_wm/results")
NO, NA = 18, 8
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        c = [6, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 4, 2, 1), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(256 * 4 * 4, n)

    def forward(self, img):                       # img : (B, 2, 3, 64, 64) uint8
        x = img.float().div(255.0).flatten(1, 2)  # (B, 6, 64, 64)
        return self.fc(self.conv(x).flatten(1))


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class PixWM(nn.Module):
    def __init__(self, kind, n):
        super().__init__()
        self.kind, self.n = kind, n
        if kind == "mamba2":
            from wm import Mamba2Block
            self.enc = Encoder(n)
            self.act_emb = nn.Linear(NA, n)
            self.blocks = nn.ModuleList([Mamba2Block(n) for _ in range(2)])
        else:
            self.enc = Encoder(n)
            if kind == "gru":
                self.cell = nn.GRU(NA, n, batch_first=True)
            elif kind == "ren":
                self.cell = ExplicitREN(NA, 1, n, n, gamma=None, alpha_bar=1.0)
            elif kind == "sren":
                self.cell = SelectiveREN(NA, n_heads=8, nx=n // 8, nq=2 * (n // 8), alpha_bar=0.9, g_min=0.02)
        self.dec = mlp(n, 256, NO)

    def states(self, img, a):
        x0 = self.enc(img)
        if self.kind == "gru":
            xs, _ = self.cell(a, x0[None].contiguous())
            return xs
        if self.kind == "ren":
            _, xs = self.cell(a, x0)
            return xs[:, 1:]
        if self.kind == "sren":
            return rollout(self.cell, a, x0, "triton" if dev == "cuda" else "torch", prec="tf32")
        # mamba2 : jeton 0 = encodage, puis un jeton par action
        sts = [b.init_state(img.shape[0]) for b in self.blocks]
        tok = x0
        new = []
        for b, st in zip(self.blocks, sts):
            st, tok = b.step((st[0].to(dev), st[1].to(dev)), tok)
            new.append(st)
        sts, feats = new, []
        for t in range(a.shape[1]):
            tok = self.act_emb(a[:, t])
            nxt = []
            for b, st in zip(self.blocks, sts):
                st, tok = b.step(st, tok)
                nxt.append(st)
            sts = nxt
            feats.append(tok)
        return torch.stack(feats, 1)

    def forward(self, img, a):
        return self.dec(self.states(img, a))


class Data:
    def __init__(self, path):
        z = np.load(path)
        O = z["Otr"].reshape(-1, NO)
        A = z["Atr"].reshape(-1, NA)
        self.mo, self.so = O.mean(0), O.std(0) + 1e-3
        self.ma, self.sa = A.mean(0), A.std(0) + 1e-3
        on = lambda O: torch.tensor((O - self.mo) / self.so, dtype=torch.float32)  # noqa: E731
        an = lambda A: torch.tensor((A - self.ma) / self.sa, dtype=torch.float32)  # noqa: E731
        self.starts = [int(s) for s in z["starts"]]
        self.tr = (on(z["Otr"]), an(z["Atr"]), torch.tensor(z["Itr"]))
        self.va = (on(z["Ova"]), an(z["Ava"]), torch.tensor(z["Iva"]))
        self.te = (on(z["Ote"]), an(z["Ate"]), torch.tensor(z["Ite"]))

    def batch(self, B, H, gen, split="tr"):
        O, A, I = self.tr if split == "tr" else self.va
        ok = [k for k, s in enumerate(self.starts) if s + 1 + H <= A.shape[1]]
        i = torch.randint(0, O.shape[0], (B,), generator=gen)
        k = torch.tensor(ok)[torch.randint(0, len(ok), (B,), generator=gen)]
        s = torch.tensor(self.starts)[k] + 1                         # état connu à s+1 (2e image)
        idx = s[:, None] + torch.arange(H + 1)
        o = O[i[:, None], idx]
        a = A[i[:, None], idx[:, :-1]]
        return I[i, k].to(dev), a.to(dev), o[:, 1:].to(dev)


@torch.no_grad()
def evaluate(model, data, horizons=(10, 50, 100, 250, 500, 1000, 1499)):
    O, A, I = data.te
    pred = model(I.to(dev), A[:, 1:].to(dev)).cpu()                  # départ à t = 1
    err = ((pred - O[:, 2:]) ** 2).mean(-1)
    res = {f"nmse@{h}": float(err[:, :h].mean()) for h in horizons}
    res["finite"] = bool(torch.isfinite(pred).all())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sren")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/data/franka_pix.npz")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = Data(a.data)
    model = PixWM(a.model, a.n).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.iters, pct_start=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    log = dict(loss=[], grad=[], val=[], nonfinite=0)
    t0 = time.time()
    for it in range(a.iters):
        img, act, o = data.batch(a.batch, a.H, gen)
        loss = ((model(img, act) - o) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            log["nonfinite"] += 1
            sched.step()
            continue
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % 50 == 0:
            log["loss"].append(loss.item())
            log["grad"].append(g.item())
        if it % 250 == 249:
            with torch.no_grad():
                vi, va_, vo = data.batch(256, a.H, torch.Generator().manual_seed(7), "va")
                log["val"].append(((model(vi, va_) - vo) ** 2).mean().item())
            print(it, log["loss"][-1], log["val"][-1], f"{time.time() - t0:.0f}s", flush=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    log["time"] = time.time() - t0
    log["sec_per_iter"] = log["time"] / a.iters
    ev = evaluate(model, data)
    name = f"pix_{a.model}_n{a.n}_H{a.H}_s{a.seed}"
    json.dump(dict(args=vars(a), nparams=nparams, log=log, eval=ev), open(f"{OUT}/{name}.json", "w"))
    torch.save(model.state_dict(), f"{OUT}/{name}.pt")
    print(name, nparams, f"{log['time'] / 60:.1f} min", {k: (round(v, 4) if isinstance(v, float) else v)
                                                       for k, v in ev.items()}, flush=True)
