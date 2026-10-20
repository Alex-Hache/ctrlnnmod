"""Analyse de l'expérience 3. Usage : python analyze_imagine.py <dossier des résultats> <dossier de sortie>."""
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
runs = [json.load(open(f)) for f in glob.glob(f"{src}/**/imagine_*.json", recursive=True)]
CLS = {"mlp_1step": "MLP 1 pas", "mlp_ms": "MLP multi-pas", "gru": "GRU", "ren": "REN contractant"}
COL = {"mlp_1step": "tab:gray", "mlp_ms": "tab:orange", "gru": "tab:blue", "ren": "tab:green"}


def cls(r):
    return r["wm"].split("_H")[0]


base = float(np.mean([r["baseline_rmse_mean"] for r in runs]))
lines = ["| World model | H imagination | runs | RMSE réelle moyenne (rad) | pire run | RMSE imaginée | réel / imaginé | gradient max |",
         "|---|---|---|---|---|---|---|---|"]
for H in (250, 50):
    for c in CLS:
        rs = [r for r in runs if cls(r) == c and r["H"] == H]
        if not rs:
            continue
        real = [r["real_rmse_mean"] for r in rs]
        imag = [r["imagined_rmse"] for r in rs]
        gmax = max(max(r["log"]["grad"]) for r in rs)
        lines.append(f"| {CLS[c]} | {H} | {len(rs)} | {np.mean(real):.3f} ({100*(np.mean(real)/base-1):+.0f} %) | "
                     f"{max(real):.3f} | {np.mean(imag):.3f} | {np.mean(np.array(real)/np.array(imag)):.2f} | {gmax:.1e} |")
lines.append(f"\nRéférence sans politique (a_t = q_ref(t+1)) : RMSE = {base:.3f} rad.")
open(f"{dst}/imagine_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
for i, c in enumerate(CLS):
    rs = [r for r in runs if cls(r) == c and r["H"] == 250]
    ys = [r["real_rmse_mean"] for r in rs]
    ax.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys)), ys, c=COL[c], s=40, zorder=3)
    ax.bar(i, np.mean(ys), color=COL[c], alpha=0.35)
ax.axhline(base, c="k", ls="--", lw=1, label="sans politique apprise")
ax.set_xticks(range(len(CLS)))
ax.set_xticklabels(list(CLS.values()))
ax.set_ylabel("RMSE de suivi sur MuJoCo (rad)")
ax.set_title("Contrôleur appris dans le world model (H = 250), déployé sur le Franka")
ax.legend()
ax = axes[1]
for name, c in (("gru_H100_n128_s1", "tab:blue"), ("gru_H100_n128_s0", "tab:cyan"),
                ("ren_H100_n128_s1", "tab:green"), ("ren_H100_n128_s0", "tab:olive")):
    r = [x for x in runs if x["wm"] == name and x["H"] == 250]
    if r:
        g = np.array(r[0]["log"]["grad"])
        ax.semilogy(np.arange(len(g)) * 10, g, c=c, lw=1, label=name.replace("_n128", ""))
ax.set_xlabel("itération")
ax.set_ylabel("norme du gradient (avant écrêtage)")
ax.set_title("Rétropropagation sur 250 pas imaginés")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{dst}/imagine.png", dpi=110)
