"""Figures et tableau de synthèse pour init_conditioning.py."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = json.load(open("experiments/init_conditioning_results.json"))
floor = R.pop("noise_floor")
SCEN = {"S1": "S1 : init sous-évaluée, γ=1.5", "S2": "S2 : init aléatoire, γ=1.5",
        "S1t": "S1t : init sous-évaluée, γ=1.05 (serré)"}
METH = {"legacy": ("ancienne init", "tab:red"), "exact": ("init exacte (Lean)", "tab:blue"),
        "barrier_1e-3": ("barrière logdet μ=1e-3", "tab:green"),
        "barrier_1e-5": ("barrière logdet μ=1e-5", "tab:olive")}

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
rows = []
for j, sc in enumerate(SCEN):
    for m, (lab, col) in METH.items():
        runs = [R[k] for k in R if k.startswith(f"{sc}/{m}/") and "error" not in R[k]]
        errs = [k for k in R if k.startswith(f"{sc}/{m}/") and "error" in R[k]]
        if not runs:
            continue
        L = min(len(r["mse"]) for r in runs)
        mse = np.array([r["mse"][:L] for r in runs])
        lam = np.array([r["lam_M"][:L] for r in runs])
        ax[0, j].semilogy(np.median(mse, 0), color=col, label=lab)
        ax[0, j].fill_between(range(L), mse.min(0), mse.max(0), color=col, alpha=.15)
        ax[1, j].semilogy(np.median(lam, 0), color=col, label=lab)
        fin = [r["mse"][-1] for r in runs]
        rows.append((sc, lab, len(runs), len(errs), np.median([r["gamma_eff"] for r in runs]),
                     np.median(fin) / floor, np.median([min(r["lam_M"]) for r in runs]),
                     max(max(r["cond_P"]) for r in runs), sum(r["nan"] for r in runs),
                     sum(r.get("rejected", 0) for r in runs),
                     np.median([r["final_hinf"] for r in runs])))
    ax[0, j].axhline(floor, ls="--", c="k", lw=.8, label="plancher de bruit")
    ax[0, j].set_title(SCEN[sc])
    ax[0, j].set_ylabel("MSE")
    ax[1, j].set_ylabel("λ_min (M_s ou −BRL)")
    ax[1, j].set_xlabel("époque")
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("experiments/init_conditioning.png", dpi=120)

hdr = ("| Scénario | Méthode | runs | erreurs | γ effectif | MSE finale / plancher | "
       "min λ_min | max cond(P) | NaN | pas rejetés | ‖G‖∞ final |")
print(hdr)
print("|" + "---|" * 11)
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.1f} | {r[6]:.1e} | "
          f"{r[7]:.1f} | {r[8]} | {r[9]} | {r[10]:.3f} |")
