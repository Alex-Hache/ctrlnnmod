"""Le κ de JJᵀ à la solution prédit-il le nombre d'itérations ? (perte ½‖H − H*‖²)"""
import glob
import json
import math
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = defaultdict(list)
for f in glob.glob("experiments/param_geometry/results/psd_*.json"):
    z = json.load(open(f))
    if z.get("task") == "psd":
        R[(z["kind"], z["target"], z["seed"])].append(z)
pts = []
for (k, t, s), runs in R.items():
    conv = [r for r in runs if r["status"] == "ok" and r["it_tol"] is not None and r.get("fp_final")
            and "kappa" in r["fp_final"]]
    if conv:
        b = min(conv, key=lambda r: r["it_tol"])
        if b["fp_final"]["kappa"] < 1e12 and b["it_tol"] > 1:
            pts.append((k, t, b["fp_final"]["kappa"], b["it_tol"]))
x = np.log10([p[2] for p in pts])
y = np.log10([p[3] for p in pts])
slope, icpt = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(5.5, 4.2))
cols = {"bm": "tab:blue", "chol": "tab:orange", "cholexp": "tab:brown", "exps": "tab:green",
        "geo_expm": "tab:red", "geo_cayley": "tab:purple"}
for k, c in cols.items():
    P = [p for p in pts if p[0] == k]
    ax.loglog([p[2] for p in P], [p[3] for p in P], "o", c=c, label=k)
kk = np.geomspace(min(p[2] for p in pts), max(p[2] for p in pts), 50)
ax.loglog(kk, 10 ** icpt * kk ** slope, "k--", lw=1, label=f"pente {slope:.2f}, r = {r:.2f}")
ax.set_xlabel("κ(JJᵀ) à la solution (empreinte de la paramétrisation)")
ax.set_ylabel("itérations jusqu'à 1e-10 (meilleur pas)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("experiments/param_geometry/analysis/kappa_vs_iters.png", dpi=120)
print(f"{len(pts)} points, pente {slope:.2f}, corrélation log-log {r:.3f}")
