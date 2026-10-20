"""Analyse de l'expérience 2. Usage : python analyze_wm.py <dossier des résultats> <dossier de sortie>."""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
runs = [json.load(open(f)) for f in glob.glob(f"{src}/**/*_H*_n*_s*.json", recursive=True)]
HOR = [10, 50, 100, 250, 500, 1000, 1500]
LAB = {"mlp_1step": "MLP 1 pas (type MBPO)", "mlp_ms": "MLP multi-pas", "gru": "GRU",
       "ren": "REN contractant (garanti)"}
COL = {"mlp_1step": "tab:gray", "mlp_ms": "tab:orange", "gru": "tab:blue", "ren": "tab:green"}

groups = {}
for r in runs:
    a = r["args"]
    groups.setdefault((a["model"], a["H"], a["n"]), []).append(r)


def agg(rs, key):
    v = np.array([r["eval"][key] for r in rs], float)
    return np.nanmean(np.where(np.isfinite(v), v, np.nan)), np.isinf(v).sum()


lines = ["| Modèle | H entr. | n | params | " + " | ".join(f"NMSE@{h}" for h in HOR) +
         " | diverg.@1500 | grad max | non-finis | temps (min) |",
         "|---|---|---|---|" + "---|" * len(HOR) + "---|---|---|---|"]
summary = {}
for (m, H, n), rs in sorted(groups.items()):
    cells = []
    for h in HOR:
        mu, ninf = agg(rs, f"nmse@{h}")
        cells.append(("∞" if np.isnan(mu) else f"{mu:.3f}") + (f" ({ninf}∞)" if ninf else ""))
    div = np.mean([r["eval"]["diverged@1500"] for r in rs])
    gmax = max(max(r["log"]["grad"]) for r in rs)
    nf = sum(r["log"]["nonfinite"] for r in rs)
    t = np.mean([r["log"]["time"] for r in rs]) / 60
    lines.append(f"| {LAB[m]} | {H} | {n} | {rs[0]['nparams']//1000}k | " + " | ".join(cells) +
                 f" | {100*div:.0f}% | {gmax:.1e} | {nf} | {t:.0f} |")
    summary[f"{m}/H{H}/n{n}"] = {f"nmse@{h}": agg(rs, f"nmse@{h}")[0] for h in HOR} | dict(
        diverged=div, grad_max=gmax, nonfinite=nf, seeds=len(rs), params=rs[0]["nparams"],
        val=float(np.mean([r["eval"]["val_nmse_250"] for r in rs])))
open(f"{dst}/table.md", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{dst}/summary.json", "w"), indent=1)
print("\n".join(lines))

# ---------------------------------------------------------------- erreur vs horizon de prédiction
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, H in zip(axes, (25, 100, 250)):
    for m in LAB:
        key = (m, H if m != "mlp_1step" else 25, 128)
        if key not in groups:
            continue
        ys = [agg(groups[key], f"nmse@{h}")[0] for h in HOR]
        ax.plot(HOR, ys, "o-", c=COL[m], label=LAB[m], ls="--" if m == "mlp_1step" else "-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"entraînement sur H = {H} pas ({H*0.02:.1f} s)")
    ax.set_xlabel("horizon de prédiction (pas de 20 ms)")
    ax.axvline(H, c="k", lw=0.6, ls=":")
axes[0].set_ylabel("NMSE moyenne sur l'horizon")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{dst}/wm_horizon.png", dpi=110)

# ---------------------------------------------------------------- passage à l'échelle en largeur
sc = glob.glob(f"{src}/**/scaling.json", recursive=True)
if sc:
    S = json.load(open(sc[0]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in (("gru", "tab:blue"), ("ren_explicit", "tab:green"), ("ren_acyclic", "tab:red")):
        ns = sorted(int(k.split("/")[1]) for k in S if k.startswith(name + "/"))
        ax.plot(ns, [S[f"{name}/{n}"]["sec_per_iter"] for n in ns], "o-", c=c,
                label={"gru": "GRU", "ren_explicit": "REN explicite (ce travail)",
                       "ren_acyclic": "REN acyclique (classique)"}[name])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("dimension d'état n (= nq)")
    ax.set_ylabel("s / itération (batch 64, H = 100, 4 vCPU)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{dst}/wm_scaling.png", dpi=110)
    print("\nScaling :")
    for k, v in S.items():
        print(f"  {k:20s} {v['sec_per_iter']:.3f} s/it  {v['params']} params")
