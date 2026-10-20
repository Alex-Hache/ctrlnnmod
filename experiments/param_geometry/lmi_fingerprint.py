"""Empreinte JJᵀ des paramétrisations LMI de ctrlnmod : application paramètres → (A, B, C).
Compare la paramétrisation PD de geotorch (décomposition propre) avec XᵀX + εI, à paramètres
aléatoires. Usage : python experiments/param_geometry/lmi_fingerprint.py"""
import json
import math
import sys

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn.utils import parametrize

import ctrlnmod.models.ssmodels.continuous.linear.h2 as h2mod
import ctrlnmod.models.ssmodels.continuous.linear.hinf as hinfmod

torch.set_default_dtype(torch.float64)


class BM(nn.Module):
    def forward(self, X):
        return X.T @ X + 1e-6 * torch.eye(X.shape[0], dtype=X.dtype)


def bm_positive_definite(module, name):
    W = getattr(module, name)
    with torch.no_grad():
        W.copy_(torch.randn_like(W) / math.sqrt(W.shape[0]) + torch.eye(W.shape[0]))
    parametrize.register_parametrization(module, name, BM(), unsafe=True)


def build(kind, nx, nu, ny, pd):
    orig = (hinfmod.geo.positive_definite, h2mod.geo.positive_definite)
    if pd == "bm":
        hinfmod.geo.positive_definite = bm_positive_definite
        h2mod.geo.positive_definite = bm_positive_definite
    try:
        if kind.startswith("hinf"):
            return hinfmod.L2BoundedLinear(nu, ny, nx, gamma=2.0, param=kind.split("_")[1])
        return h2mod.H2Linear(nu, ny, nx, gamma_2=2.0)
    finally:
        hinfmod.geo.positive_definite, h2mod.geo.positive_definite = orig


def fingerprint(model):
    names, vals = zip(*[(k, v.detach().clone()) for k, v in model.named_parameters()])

    def abc(*v):
        A, B, C = functional_call(model, dict(zip(names, v)), (), {}, strict=False) if False else (None,) * 3
        return A

    class Wrap(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s):
            return torch.cat([t.reshape(-1) for t in s.m._frame()])
    w = Wrap(model)
    full = ["m." + k for k in names]

    def f(*v):
        return functional_call(w, dict(zip(full, v)), ())
    Js = torch.autograd.functional.jacobian(f, vals)
    J = torch.cat([j.reshape(j.shape[0], -1) for j in Js], 1)
    ev = torch.linalg.eigvalsh(J @ J.T)
    tol = 1e-10 * ev[-1]
    pos = ev[ev > tol]
    return dict(rank=int(pos.numel()), dim=int(ev.numel()), kappa=float(ev[-1] / ev[0].clamp(min=1e-300)),
                kappa_range=float(pos[-1] / pos[0]))


res = []
for kind in ("hinf_sqrtm", "hinf_riccati", "h2"):
    for pd in ("geotorch", "bm"):
        for nx in (2, 4, 8):
            for seed in range(5):
                torch.manual_seed(seed)
                try:
                    fp = fingerprint(build(kind, nx, 2, 2, pd))
                    res.append(dict(kind=kind, pd=pd, nx=nx, seed=seed, **fp))
                except Exception as e:                       # noqa: BLE001
                    res.append(dict(kind=kind, pd=pd, nx=nx, seed=seed, err=repr(e)[:200]))
                    print(kind, pd, nx, seed, repr(e)[:200], file=sys.stderr)

out = "experiments/param_geometry/analysis"
json.dump(res, open(f"{out}/lmi_fingerprint.json", "w"), indent=1)
lines = ["| Modèle LMI | PD | nx | rang / dim | κ(JJᵀ) médian | κ sur l'image (médian) |", "|---|---|---|---|---|---|"]
for kind in ("hinf_sqrtm", "hinf_riccati", "h2"):
    for pd in ("geotorch", "bm"):
        for nx in (2, 4, 8):
            R = [r for r in res if (r["kind"], r["pd"], r["nx"]) == (kind, pd, nx) and "rank" in r]
            if not R:
                lines.append(f"| {kind} | {pd} | {nx} | échec | | |")
                continue
            lines.append(f"| {kind} | {pd} | {nx} | {min(r['rank'] for r in R)} / {R[0]['dim']} | "
                         f"{np.median([r['kappa'] for r in R]):.1e} | {np.median([r['kappa_range'] for r in R]):.1e} |")
txt = "\n".join(lines) + "\n"
open(f"{out}/lmi_fingerprint.md", "w").write(txt)
print(txt)
