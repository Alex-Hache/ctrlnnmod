"""Analyse de l'étude maître–élève : meilleur pas par méthode, écarts, diagnostics, courbes.

Usage : python analyze_ts.py <dossier des JSON> [dossier de sortie]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/optim/results"
out = sys.argv[2] if len(sys.argv) > 2 else "experiments/optim/analysis"
os.makedirs(out, exist_ok=True)
R = defaultdict(list)                                  # (rho, seed, method) -> runs
for f in glob.glob(f"{src}/**/ts_*.json", recursive=True):
    z = json.load(open(f))
    R[(z["rho"], z["seed"], z["method"])].append(z)
rhos = sorted({k[0] for k in R})
seeds = sorted({k[1] for k in R})
METH = ["free_gd", "ren_gd", "riem_gd", "free_adam", "ren_adam", "riem_adam"]
LAB = {"free_gd": "libre, GD", "ren_gd": "REN, GD sur θ", "riem_gd": "REN, GD riemannien",
       "free_adam": "libre, Adam", "ren_adam": "REN, Adam sur θ", "riem_adam": "REN, Adam riemannien"}


def best(runs):
    ok = [r for r in runs if r["status"] == "ok" and r["final_train"] is not None and np.isfinite(r["final_train"])]
    return min(ok, key=lambda r: r["final_train"]) if ok else None


def iters_to(r, thr):
    for it, v in zip(r["log"]["it"], r["log"]["train"]):
        if v <= thr:
            return it
    return np.inf


rows, summ = [], {}
for rho in rhos:
    tm = None
    for m in METH:
        B = [best(R[(rho, s, m)]) for s in seeds]
        B = [b for b in B if b is not None]
        if not B:
            continue
        tm = B[0]["teacher"]["margin"]
        tr = np.exp(np.mean([np.log(b["final_train"]) for b in B]))
        te = np.exp(np.mean([np.log(b["final_test"]) for b in B]))
        it3 = np.median([iters_to(b, 1e-2) for b in B])
        cM = np.exp(np.mean([np.log(b["log"]["cond_M"][-1]) for b in B])) if B[0]["log"]["cond_M"] else np.nan
        cJ = np.exp(np.mean([np.log(b["log"]["cond_JJt"][-1]) for b in B])) if B[0]["log"]["cond_JJt"] else np.nan
        lrs = sorted({b["lr"] for b in B})
        summ[f"{rho}/{m}"] = dict(train=tr, test=te, iters_1e2=float(it3), cond_M=cM, cond_JJt=cJ, lrs=lrs,
                                  teacher_margin=tm, seeds=len(B))
        rows.append(f"| {rho} | {tm:.4f} | {LAB[m]} | {tr:.2e} | {te:.2e} | {it3:.0f} | "
                    f"{'—' if np.isnan(cM) else f'{cM:.1e}'} | {'—' if np.isnan(cJ) else f'{cJ:.1e}'} | "
                    f"{', '.join(str(v) for v in lrs)} |")
head = ("| ρ maître | marge maître | élève | perte finale (entr.) | perte test | itér. jusqu'à 1e-2 | "
        "cond(XᵀX) final | cond(JJᵀ) final | meilleur pas |\n|---|---|---|---|---|---|---|---|---|")
txt = head + "\n" + "\n".join(rows) + "\n"
# écarts
gap = ["\n| ρ | REN/libre (GD) | riem/libre (GD) | REN/libre (Adam) | riem/libre (Adam) |", "|---|---|---|---|---|"]
for rho in rhos:
    g = lambda a, b: summ.get(f"{rho}/{a}", {}).get("train", np.nan) / summ.get(f"{rho}/{b}", {}).get("train", np.nan)  # noqa: E731
    gap.append(f"| {rho} | {g('ren_gd', 'free_gd'):.1f}× | {g('riem_gd', 'free_gd'):.1f}× | "
               f"{g('ren_adam', 'free_adam'):.1f}× | {g('riem_adam', 'free_adam'):.1f}× |")
txt += "\n".join(gap) + "\n"
open(f"{out}/table_ts.md", "w").write(txt)
json.dump(summ, open(f"{out}/summary_ts.json", "w"), indent=1, default=float)
print(txt)

fig, axes = plt.subplots(2, len(rhos), figsize=(4.2 * len(rhos), 7), sharey="row")
col = {"free": "tab:gray", "ren": "tab:red", "riem": "tab:blue"}
for j, rho in enumerate(rhos):
    for i, opt in enumerate(("gd", "adam")):
        ax = axes[i, j]
        for kind in ("free", "ren", "riem"):
            m = f"{kind}_{opt}"
            for s in seeds:
                b = best(R[(rho, s, m)])
                if b:
                    ax.semilogy(b["log"]["it"], b["log"]["train"], c=col[kind], lw=1.2, alpha=0.8,
                                label=LAB[m] if s == seeds[0] else None)
        ax.set_title(f"ρ = {rho} ({opt.upper()})")
        ax.set_xlabel("itération")
        if j == 0:
            ax.set_ylabel("perte d'entraînement (NMSE)")
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out}/curves_ts.png", dpi=110)
print("figure", f"{out}/curves_ts.png")
