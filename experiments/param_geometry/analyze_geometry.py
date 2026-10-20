"""Tableaux : itérations de descente de gradient pure jusqu'à 1e-10 (meilleur pas), empreinte JJᵀ.
Usage : python analyze_geometry.py <dossier JSON>"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/param_geometry/results"
out = "experiments/param_geometry/analysis"
os.makedirs(out, exist_ok=True)
R = defaultdict(list)
for f in glob.glob(f"{src}/**/*.json", recursive=True):
    z = json.load(open(f))
    if "task" not in z:
        continue
    key = (z["task"], z["kind"], z.get("target", z.get("phi_frac")), z["seed"])
    R[key].append(z)


def best(runs):
    ok = [r for r in runs if r["status"] == "ok"]
    if not ok:
        return None
    conv = [r for r in ok if r["it_tol"] is not None]
    return min(conv, key=lambda r: r["it_tol"]) if conv else min(ok, key=lambda r: r["final"])


def cell(task, kind, tgt):
    B = [best(R[(task, kind, tgt, s)]) for s in (0, 1)]
    B = [b for b in B if b]
    if not B:
        return "—", None
    its = [b["it_tol"] for b in B]
    if all(i is not None for i in its):
        return f"{int(np.median(its))}", B
    fin = max(b["final"] for b in B)
    return f"> 20 000 ({fin:.0e})", B


LAB = {"euclid": "H libre + projection", "bm": "XᵀX + εI (ctrlnnmod, REN)", "chol": "LLᵀ + εI (Cholesky)",
       "cholexp": "Cholesky, diag. exp", "exps": "exp(S)", "geo_expm": "geotorch PD, Q par expm",
       "geo_cayley": "geotorch PD, Q par Cayley", "polar": "Q libre + rétraction polaire"}
lines = ["## Matrices définies positives (n = 6, départ H₀ = I, perte ½‖H − H*‖²)\n",
         "Itérations jusqu'à ‖H − H*‖²/‖H*‖² < 1e-10 (médiane sur 2 graines, meilleur pas). "
         "Empreinte de JJᵀ au départ : rang / 21 et κ.\n",
         "| Paramétrisation | rang JJᵀ en I | κ en I | κ bien cond. | bien conditionnée (κ=10) | mal conditionnée (κ=10³) | "
         "valeurs propres groupées | près du bord (10⁻⁴…10⁻¹) |", "|---|---|---|---|---|---|---|---|"]
for k in ("euclid", "bm", "chol", "cholexp", "exps", "geo_expm", "geo_cayley"):
    cs = [cell("psd", k, t) for t in ("wellcond", "illcond", "clustered", "nearbdry")]
    fp0 = next((b[0]["fp_init"] for _, b in cs if b), None)
    fpw = next((b[0]["fp_final"] for c, b in [cs[0]] if b and b[0].get("fp_final")), None)
    k0 = "—" if not fp0 else (f"{fp0['kappa']:.1f}" if fp0["kappa"] < 1e12 else "∞")
    kw = "—" if not fpw or "kappa" not in fpw else (f"{fpw['kappa']:.0f}" if fpw["kappa"] < 1e12 else "∞")
    lines.append(f"| {LAB[k]} | {fp0['rank'] if fp0 else '—'} / 21 | {k0} | {kw} | " + " | ".join(c for c, _ in cs) + " |")
lines += ["\n## Même chose, perte KL : tr(H H*⁻¹) − log det(H H*⁻¹) − n\n",
          "| Paramétrisation | bien conditionnée | mal conditionnée | valeurs propres groupées | près du bord |",
          "|---|---|---|---|---|"]
for k in ("euclid", "bm", "chol", "cholexp", "exps", "geo_expm", "geo_cayley"):
    lines.append(f"| {LAB[k]} | " + " | ".join(cell("psdkl", k, t)[0] for t in ("wellcond", "illcond", "clustered", "nearbdry")) + " |")
lines += ["\n## Matrices orthogonales (SO(4), deux plans tournés de φ*, départ Q = I)\n",
          "| Paramétrisation | φ* = 0,1π | 0,5π | 0,8π | 0,95π |", "|---|---|---|---|---|"]
for k in ("geo_expm", "geo_cayley", "polar"):
    lines.append(f"| {LAB[k]} | " + " | ".join(cell("orth", k, p)[0] for p in (0.1, 0.5, 0.8, 0.95)) + " |")
lines.append("\nPrédiction Cayley (Lean `cayley_induced_flow`) : vitesse angulaire × cos⁴(φ/2) ; "
             "cos⁴(φ/2) vaut " + ", ".join(f"{math.cos(p * math.pi / 2) ** 4:.1e}" for p in (0.1, 0.5, 0.8, 0.95))
             + " pour φ = 0,1π ; 0,5π ; 0,8π ; 0,95π.")
txt = "\n".join(lines) + "\n"
open(f"{out}/geometry.md", "w").write(txt)
print(txt)
