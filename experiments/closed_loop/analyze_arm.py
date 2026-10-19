"""Analyse de l'expérience 1. Usage : python analyze_arm.py <dossier des résultats> <dossier de sortie>."""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arm_boosting import OBST_C, OBST_R, Q0, QSTAR, link_points, rollout, sample_x0  # noqa: E402
from run_arm import make  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
runs = [json.load(open(f)) for f in glob.glob(f"{src}/**/*_s*.json", recursive=True)
        if "random" not in f]
SUITES = ["train_3s", "long_30s", "wide_ic_30s", "pushes_30s", "payload_x1.5_30s",
          "payload_x2.0_30s", "payload_x3.0_30s"]
LAB = {"mlp": "MLP libre (retour d'état)", "pb_rnn": "PB + RNN libre", "pb_ren": "PB + REN (γ garanti)"}

# ---------------------------------------------------------------------------- tableau
lines = ["| Politique | γ | seeds | " + " | ".join(SUITES) + " | divergences (toutes suites) |",
         "|---|---|---|" + "---|" * len(SUITES) + "---|"]
groups = {}
for r in runs:
    groups.setdefault((r["kind"], r["gamma"] if r["kind"] == "pb_ren" else None), []).append(r)
summary = {}
for (kind, g), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
    row = {}
    for s in SUITES:
        succ = [r["eval"][s]["success"] for r in rs]
        row[s] = (np.mean(succ), np.min(succ), np.max(succ))
    div = np.mean([r["eval"][s]["diverged"] for r in rs for s in SUITES])
    nonfinite = sum(r["nonfinite"] for r in rs)
    summary[f"{kind}/{g}"] = dict(rows={s: row[s][0] for s in SUITES}, diverged=div, nonfinite=nonfinite,
                                  final_loss=float(np.mean([r["hist"][-1] for r in rs])), n=len(rs))
    cells = [f"{100*m:.0f}% ({100*lo:.0f}–{100*hi:.0f})" for (m, lo, hi) in (row[s] for s in SUITES)]
    lines.append(f"| {LAB[kind]} | {g if g else '—'} | {len(rs)} | " + " | ".join(cells) + f" | {100*div:.1f}% |")
open(f"{dst}/table.md", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{dst}/summary.json", "w"), indent=1)
print("\n".join(lines))

# ---------------------------------------------------------------------------- paramètres aléatoires
rp = glob.glob(f"{src}/**/random_parameters.json", recursive=True)
if rp:
    R = json.load(open(rp[0]))
    print("\nParamètres aléatoires (30 s, 16 conditions initiales par tirage) :")
    for k, v in R.items():
        print(f"  {k:22s} atteint la cible : {100*v['reached']:.1f}%  "
              f"tirages 100% stables : {100*v['all_reached']:.1f}%  divergence : {100*v['diverged']:.1f}%")

# ---------------------------------------------------------------------------- figure : trajectoires
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
torch.manual_seed(0)
x0 = sample_x0(12, spread=0.15, gen=torch.Generator().manual_seed(77))
for ax, kind in zip(axes, ("mlp", "pb_rnn", "pb_ren")):
    cand = [r for r in runs if r["kind"] == kind and r["gamma"] == 200.0]
    if not cand:
        continue
    best = min(cand, key=lambda r: r["hist"][-1])          # graine de perte finale minimale
    f = sorted(glob.glob(f"{src}/**/{kind}_g200_s{best['seed']}.pt", recursive=True))
    pol = make(kind)
    pol.load_state_dict(torch.load(f[0]))
    with torch.no_grad():
        xs, _ = rollout(pol, x0, 1500)
    ee = link_points(xs[..., :2])[..., -1, :]
    for i in range(x0.shape[0]):
        ax.plot(ee[i, :150, 0], ee[i, :150, 1], c="tab:blue", lw=0.8)
        ax.plot(ee[i, 150:, 0], ee[i, 150:, 1], c="tab:red", lw=0.8, alpha=0.7)
    ax.add_patch(plt.Circle(OBST_C.tolist(), OBST_R, color="k", alpha=0.3))
    for q, c in ((Q0, "tab:green"), (QSTAR, "tab:purple")):
        pts = link_points(q[None])[0]
        ax.plot(pts[:, 0], pts[:, 1], c=c, lw=3)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.4, 0.6)
    ax.set_title(f"{LAB[kind]} (meilleure graine : {best['seed']})\nbleu : 0–3 s (entraînement), rouge : 3–30 s")
fig.tight_layout()
fig.savefig(f"{dst}/arm_trajectories.png", dpi=110)

# ---------------------------------------------------------------------------- figure : succès par suite
fig, ax = plt.subplots(figsize=(12, 4))
keys = [k for k in summary if k.split("/")[0] in ("mlp", "pb_rnn") or k == "pb_ren/200.0"]
w = 0.8 / max(len(keys), 1)
for i, k in enumerate(keys):
    vals = [100 * summary[k]["rows"][s] for s in SUITES]
    ax.bar(np.arange(len(SUITES)) + i * w, vals, w, label=LAB[k.split("/")[0]])
ax.set_xticks(np.arange(len(SUITES)) + w * (len(keys) - 1) / 2)
ax.set_xticklabels(["entraînement\n3 s", "même init.\n30 s", "init. larges\n30 s", "poussées\n30 s",
                    "charge ×1,5", "charge ×2", "charge ×3"])
ax.set_ylabel("succès (%) : cible atteinte, sans collision")
ax.legend()
fig.tight_layout()
fig.savefig(f"{dst}/arm_success.png", dpi=110)
