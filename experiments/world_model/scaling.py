"""
Passage à l'échelle en largeur : temps d'une itération d'entraînement (forward + backward,
batch 64, H = 100) en fonction de la dimension d'état n (nq = n).

- gru            : torch.nn.GRUCell (référence non contrainte, noyau optimisé)
- ren_explicit   : ExplicitREN (D11 = 0, Λ par dominance diagonale) — ce travail
- ren_acyclic    : même paramétrisation, mais D11 = -tril(H22, -1) et résolution séquentielle
                   neurone par neurone (REN « acyclique » classique), frame calculé une fois
"""
import json
import os
import statistics
import time

import torch

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
from ctrlnmod.models.discrete.selective_ren import SelectiveREN
from wm import Mamba2Block

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
OUT = os.environ.get("OUT_DIR", "experiments/world_model/results")
ONLY = os.environ.get("ONLY", "")
B, H, NA = 64, 100, 8


def run_gru(n):
    cell = torch.nn.GRUCell(NA, n)
    def it(a):
        x = torch.zeros(B, n)
        s = 0
        for t in range(H):
            x = cell(a[:, t], x)
            s = s + x.square().mean()
        s.backward()
    return cell, it


def run_ren(n, acyclic=False):
    ren = ExplicitREN(NA, 1, n, n)
    def it(a):
        fr = ren.frame()
        x = torch.zeros(B, n)
        s = 0
        if acyclic:
            Hm = ren.H()
            H22 = Hm[n:2 * n, n:2 * n]
            lam = 0.5 * torch.diagonal(H22)
            D11 = -torch.tril(H22, -1)
        for t in range(H):
            if not acyclic:
                x, _ = ren.step(fr, x, a[:, t])
            else:
                base = x @ fr["C1"].T + a[:, t] @ ren.D12.T
                ws = []
                for k in range(n):
                    vk = base[:, k] + (torch.stack(ws, 1) @ D11[k, :k] if k else 0.0)
                    ws.append(torch.tanh(vk / lam[k]))
                w = torch.stack(ws, 1)
                x = x @ fr["A"].T + w @ fr["Bw"].T + a[:, t] @ fr["Bu"].T
            s = s + x.square().mean()
        s.backward()
    return ren, it


def run_sren(n):
    m = SelectiveREN(NA, n_heads=8, nx=n // 8, nq=2 * (n // 8))
    def it(a):
        fr = m.frame()
        x = torch.zeros(B, m.n_state)
        s = 0
        for t in range(H):
            x = m.step(fr, x, a[:, t])
            s = s + x.square().mean()
        s.backward()
    return m, it


def run_mamba(n):
    """n = d_model (état SSM : 2n · 16 par couche)."""
    emb, blk = torch.nn.Linear(NA, n), Mamba2Block(n)
    mod = torch.nn.ModuleList([emb, blk])
    def it(a):
        st = blk.init_state(B)
        s = 0
        for t in range(H):
            st, y = blk.step(st, emb(a[:, t]))
            s = s + y.square().mean()
        s.backward()
    return mod, it


def bench(fn, n, reps=3):
    torch.manual_seed(0)
    mod, it = fn(n)
    a = torch.randn(B, H, NA)
    it(a)  # échauffement
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        it(a)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), sum(p.numel() for p in mod.parameters())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for n in (32, 64, 128, 256, 512, 1024):
        for name, fn in (("gru", run_gru), ("ren_explicit", run_ren), ("sren", run_sren),
                         ("mamba2", run_mamba), ("ren_acyclic", lambda n: run_ren(n, acyclic=True))):
            if name == "ren_acyclic" and n > 256:
                continue
            if ONLY and name not in ONLY.split(","):
                continue
            t, p = bench(fn, n)
            res[f"{name}/{n}"] = dict(sec_per_iter=t, params=p)
            print(name, n, f"{t:.3f}s", p, flush=True)
            json.dump(res, open(f"{OUT}/scaling.json", "w"), indent=1)
