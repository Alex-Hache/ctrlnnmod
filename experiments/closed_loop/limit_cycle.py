"""Erreur de modèle (charge ×k) : cycle limite du MLP entraîné, convergence du PB-REN.
Usage : python limit_cycle.py <dossier des résultats> <dossier de sortie>"""
import glob
import json
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arm_boosting import NOMINAL, QSTAR, rollout, sample_x0  # noqa: E402
from run_arm import make  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
T = 6000  # 120 s
x0 = sample_x0(50, gen=torch.Generator().manual_seed(4242))
# graines qui évitent l'obstacle (perte finale basse) pour chaque politique
models = [("mlp", 200.0, s) for s in (0, 1)] + [("pb_ren", 200.0, s) for s in (1, 2)] + \
         [("pb_ren", 1000.0, s) for s in (0, 1)]
res, curves = {}, {}
for kind, g, s in models:
    f = glob.glob(f"{src}/**/{kind}_g{g:g}_s{s}.pt", recursive=True)[0]
    pol = make(kind, gamma=g)
    pol.load_state_dict(torch.load(f))
    for k in (1.0, 1.5, 2.0, 3.0):
        with torch.no_grad():
            xs, _ = rollout(pol, x0, T, dict(NOMINAL, m2=NOMINAL["m2"] * k))
        last = xs[:, -1000:]
        amp = (last[..., :2].amax(1) - last[..., :2].amin(1)).norm(dim=-1)
        res[f"{kind}_g{g:g}_s{s}/x{k}"] = dict(amp_median=float(amp.median()), amp_max=float(amp.max()),
                                              speed_median=float(last[..., 2:].norm(dim=-1).mean(1).median()))
        if s == min(m[2] for m in models if m[0] == kind and m[1] == g):
            curves[(kind, g, k)] = xs[0, :, 1].numpy()
json.dump(res, open(f"{dst}/limit_cycle.json", "w"), indent=1)
for k, v in res.items():
    print(k, {a: round(b, 3) for a, b in v.items()})

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
t = torch.arange(T + 1) * 0.02
for ax, k in zip(axes, (1.0, 2.0, 3.0)):
    ax.plot(t, curves[("mlp", 200.0, k)], c="tab:orange", lw=0.8, label="MLP libre (entraîné)")
    ax.plot(t, curves[("pb_ren", 200.0, k)], c="tab:green", lw=1.2, label="PB + REN (entraîné)")
    ax.axhline(float(QSTAR[1]), c="k", lw=0.6, ls=":")
    ax.set_title(f"charge du 2e segment × {k:g}")
    ax.set_xlabel("temps (s)")
axes[0].set_ylabel("q₂ (rad)")
axes[0].legend()
fig.tight_layout()
fig.savefig(f"{dst}/arm_limit_cycle.png", dpi=110)
