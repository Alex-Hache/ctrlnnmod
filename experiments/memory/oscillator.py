"""
Test de la limite structurelle de la classe ℓ∞ : les oscillations faiblement amorties.

Maître : 4 oscillateurs indépendants x_i⁺ = 0,99 R(θ_i) x_i + B_i u, θ ∈ {3°, 10°, 25°, 45°},
y = C x⁺ (2 sorties), entrée lisse à 2 voies. Une rotation de θ a une somme de lignes
0,99(|cos θ| + |sin θ|) > 0,99 dès que θ > 0 : la classe ℓ∞ (portes par unité) ne peut pas la
représenter ; la classe par blocs de 2 (norme mixte max ℓ²) le peut.
Élèves : linf, block, ren, free, gru (16 états). Perte : erreur quadratique sur toute la sortie.
Usage : python oscillator.py --sweep
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

sys.path.insert(0, os.path.dirname(__file__))
from memory_tasks import BlockOut, FreeREN, GRUm, LinfGated, RENm  # noqa: E402

OUT = os.environ.get("OUT_DIR", "experiments/memory/results_osc")
TH = (3, 10, 25, 45)


def teacher(seed):
    g = torch.Generator().manual_seed(500 + seed)
    A = torch.zeros(8, 8)
    for i, th in enumerate(TH):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 0.99 * torch.tensor([[c, -s], [s, c]])
    return A, torch.randn(8, 2, generator=g) * 0.3, torch.randn(2, 8, generator=g) / math.sqrt(8)


def inputs(n, T, seed):
    g = torch.Generator().manual_seed(seed)
    e, u = torch.randn(n, T, 2, generator=g), torch.zeros(n, T, 2)
    for t in range(T):
        u[:, t] = (0.7 * u[:, t - 1] if t else 0) + 0.71 * e[:, t]
    return u


def sim(A, B, C, u):
    x, ys = torch.zeros(u.shape[0], 8), []
    for t in range(u.shape[1]):
        x = x @ A.T + u[:, t] @ B.T
        ys.append(x @ C.T)
    return torch.stack(ys, 1)


def make(kind, n=16):
    return {"linf": lambda: LinfGated2(n), "block": lambda: BlockOut(2, n, k=2, ny=2), "ren": lambda: RENy(n),
            "free": lambda: FreeY(n), "gru": lambda: GRUy(n),
            "linfp": lambda: LinfP(n), "blockp": lambda: BlockP(n),
            "blockp_nogate": lambda: BlockP(n, ungated=n // 2), "blockp_half": lambda: BlockP(n, ungated=n // 4),
            "blockh": lambda: BlockP(n, norm="homo"), "blockh_half": lambda: BlockP(n, ungated=n // 4, norm="homo"),
            "linfh": lambda: LinfP(n, norm="homo")}[kind]()


class LinfP(torch.nn.Module):
    """Classe ℓ∞ avec projection radiale (bord atteignable) et ρ = 0,999."""

    def __init__(self, n, norm="proj"):
        super().__init__()
        from ctrlnmod.models.discrete.linf_gated import LinfGatedRNN
        self.m, self.C = LinfGatedRNN(2, n, rho=0.999, norm=norm), torch.nn.Linear(n, 2)

    def forward(self, u):
        return self.C(self.m(u))


class BlockP(torch.nn.Module):
    """Blocs de 2 avec projection radiale et ρ = 0,999."""

    def __init__(self, n, ungated=0, norm="proj"):
        super().__init__()
        from ctrlnmod.models.discrete.linf_gated import BlockGatedRNN
        self.m = BlockGatedRNN(2, n // 2, k=2, rho=0.999, norm=norm, ungated_heads=ungated)
        self.C = torch.nn.Linear(n, 2)

    def forward(self, u):
        return self.C(self.m(u))


class LinfGated2(LinfGated):
    def __init__(self, n):
        super().__init__(2, n)
        self.C = torch.nn.Linear(n, 2)


class RENy(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
        self.m = ExplicitREN(2, 2, n, 2 * n, gamma=None, alpha_bar=1.0)

    def forward(self, u):
        return self.m(u)[0]


class FreeY(FreeREN):
    def __init__(self, n):
        torch.nn.Module.__init__(self)
        from ctrlnmod.models.discrete.explicit_ren import ExplicitREN
        ren = ExplicitREN(2, 2, n, 2 * n, gamma=None, alpha_bar=1.0)
        fr = {k: v.detach().clone() for k, v in ren.frame().items()}
        for k in ("A", "Bw", "Bu", "C1", "bx"):
            setattr(self, k, torch.nn.Parameter(fr[k]))
        self.log_lam = torch.nn.Parameter(fr["Lambda"].clamp(min=1e-3).log())
        for k in ("D12", "C2", "D21", "D22", "bv", "by"):
            setattr(self, k, torch.nn.Parameter(getattr(ren, k).detach().clone()))


class GRUy(GRUm):
    def __init__(self, n):
        super().__init__(2, n)
        self.C = torch.nn.Linear(n, 2)


def run(kind, lr, seed, iters=2000):
    torch.manual_seed(seed)
    A, B, C = teacher(seed)
    u_tr, u_te = inputs(256, 100, 10 + seed), inputs(128, 300, 20 + seed)
    y_tr, y_te = sim(A, B, C, u_tr), sim(A, B, C, u_te)
    sc = y_tr.std()
    y_tr, y_te = y_tr / sc, y_te / sc
    m = make(kind)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0, status, log = time.time(), "ok", []
    for it in range(iters):
        loss = ((m(u_tr) - y_tr) ** 2).mean()
        if not torch.isfinite(loss):
            status = f"diverged@{it}"
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if it % 50 == 0:
            log.append(float(loss.detach()))
    with torch.no_grad():
        test = float(((m(u_te) - y_te) ** 2).mean())
    return dict(kind=kind, lr=lr, seed=seed, status=status, test=test, train=log[-1] if log else None,
                log=log, time=time.time() - t0)


def _job(a):
    torch.set_num_threads(1)
    r = run(*a)
    json.dump(r, open(f"{OUT}/osc_{a[0]}_s{a[2]}_lr{a[1]}.json", "w"))
    return f"{a}: {r['status']} test={r['test']:.4f} {r['time']:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ap2 = os.environ.get("OSC_KINDS", "linf,block,ren,free,gru").split(",")
    jobs = list(itertools.product(ap2, (1e-3, 3e-3, 1e-2), (0, 1, 2)))
    from tqdm import tqdm
    with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        for line in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc="oscillateurs", mininterval=10):
            tqdm.write(line)
