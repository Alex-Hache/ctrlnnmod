"""Analyse des expériences de mémoire certifiée.

M1 : erreur mesurée du REN contre le plancher prédit (1 − γᾱᵏ)²₊, par largeur (le théorème dit :
     même plancher pour toutes les largeurs). M2 : rappel sélectif, par modèle et délai.
Usage : python analyze_memory.py <dossier JSON> [sortie]
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/memory/results"
out = sys.argv[2] if len(sys.argv) > 2 else "experiments/memory/analysis"
os.makedirs(out, exist_ok=True)
R = defaultdict(list)
for f in glob.glob(f"{src}/**/m[12]_*.json", recursive=True):
    z = json.load(open(f))
    R[(z["task"], z["kind"], z["n"], z["lag"], z["gamma"], z["ab"], z["seed"])].append(z)


def best(runs):
    ok = [r for r in runs if r["status"] == "ok" and np.isfinite(r["test"])]
    return min(ok, key=lambda r: r["test"]) if ok else None


def agg(task, kind, n, lag, gamma, ab):
    B = [best(R[(task, kind, n, lag, gamma, ab, s)]) for s in (0, 1, 2)]
    B = [b for b in B if b]
    return (np.mean([b["test"] for b in B]), B) if B else (np.nan, [])


# ------------------------------------------------------------------ M1
lags = (2, 4, 8, 16, 32, 64)
lines = ["## M1 — rappel retardé (γ = 4)\n",
         "Erreur de test (meilleur pas, moyenne sur les graines) ; entre crochets : plancher prédit "
         "(1 − γᾱᵏ)²₊ ; k* = ln γ / ln(1/ᾱ).\n"]
viol = 0
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, ab in zip(axes, (0.8, 0.9, 0.95)):
    ks = math.log(4) / math.log(1 / ab)
    lines.append(f"\n**ᾱ = {ab}, k* = {ks:.1f}**\n\n| largeur | " + " | ".join(f"k={k}" for k in lags) + " |\n" + "|---" * (len(lags) + 1) + "|")
    for n, c in zip((8, 32, 128), ("#9ecae1", "#3182bd", "#08306b")):
        row, ys = [], []
        for k in lags:
            e, B = agg("m1", "ren", n, k, 4.0, ab)
            fl = max(0.0, 1 - 4 * ab ** k) ** 2
            for b in B:                                   # vérification du théorème, run par run
                if b["test"] < fl - 0.02:
                    viol += 1
            row.append(f"{e:.3f} [{fl:.2f}]")
            ys.append(e)
        lines.append(f"| {n} | " + " | ".join(row) + " |")
        ax.semilogx(lags, ys, "o-", c=c, label=f"REN, largeur {n}")
    kk = np.geomspace(2, 64, 200)
    ax.semilogx(kk, np.maximum(0, 1 - 4 * ab ** kk) ** 2, "k--", lw=1.5, label="plancher prédit (Lean)")
    for kd, c in (("free", "tab:gray"), ("gru", "tab:green")):
        ys = [agg("m1", kd, 32, k, None, 1.0)[0] for k in lags]
        ax.semilogx(lags, ys, "s:", c=c, ms=4, label={"free": "libre (mêmes équations)", "gru": "GRU"}[kd])
    ax.axvline(ks, c="tab:red", lw=0.8)
    ax.set_title(f"ᾱ = {ab}, γ = 4 : k* = {ks:.1f}")
    ax.set_xlabel("délai k")
    ax.set_xticks(lags)
    ax.set_xticklabels([str(k) for k in lags])
    ax.minorticks_off()
axes[0].set_ylabel("erreur quadratique de rappel")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out}/m1_threshold.png", dpi=120)
lines.append(f"\nRuns sous le plancher prédit (violations du théorème, tolérance 0,02) : **{viol}**.\n")
lines.append("\nContrôles (largeur 32, sans certificat) :\n\n| modèle | " + " | ".join(f"k={k}" for k in lags) + " |\n" + "|---" * (len(lags) + 1) + "|")
for kd in ("free", "gru"):
    lines.append(f"| {kd} | " + " | ".join(f"{agg('m1', kd, 32, k, None, 1.0)[0]:.3f}" for k in lags) + " |")

# ------------------------------------------------------------------ M2
LAB = {"ren": "REN certifié (sans porte)", "free": "mêmes équations, libres", "lru": "LRU (stable, sans porte)",
       "sren": "REN sélectif (porte par tête)", "linf": "RNN à portes par unité, certifié ℓ∞ (nouveau)",
       "gru": "GRU", "mamba2": "Mamba-2"}
lines.append("\n## M2 — rappel sélectif avec distracteurs (largeur 32)\n\nErreur de test (1 = ne rien rappeler).\n\n"
             "| modèle | K = 16 | K = 64 |\n|---|---|---|")
for kd in LAB:
    vals = [agg("m2", kd, 32, K, None, 1.0)[0] for K in (16, 64)]
    lines.append(f"| {LAB[kd]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
txt = "\n".join(lines) + "\n"
open(f"{out}/memory.md", "w").write(txt)
print(txt)
