"""Tableau du test des oscillateurs. Usage : python analyze_osc.py <dossier JSON>"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/memory/results_osc"
R = defaultdict(list)
for f in glob.glob(f"{src}/**/osc_*.json", recursive=True):
    z = json.load(open(f))
    R[(z["kind"], z["seed"])].append(z)
LAB = {"linf": "ℓ∞, porte par unité (certifié)", "block": "blocs de 2, porte par paire (certifié)",
       "linfp": "ℓ∞, projection radiale, ρ = 0,999", "blockp": "blocs de 2, projection radiale, ρ = 0,999",
       "blockp_nogate": "blocs de 2, projection, sans porte (z ≡ 1)",
       "blockp_half": "blocs de 2, projection, moitié des têtes sans porte",
       "linfh": "ℓ∞, forme homogène 1/(r + softplus(c))",
       "blockh": "blocs de 2, forme homogène",
       "blockh_half": "blocs de 2, forme homogène, moitié sans porte",
       "ren": "REN (certifié)", "free": "mêmes équations, libres", "gru": "GRU"}
rows = ["| Élève (16 états) | erreur de test (moyenne) | pire graine |", "|---|---|---|"]
for k in LAB:
    B = []
    for s in sorted({key[1] for key in R}):
        ok = [r for r in R[(k, s)] if r["status"] == "ok"]
        if ok:
            B.append(min(ok, key=lambda r: r["test"])["test"])
    if B:
        rows.append(f"| {LAB[k]} | {np.mean(B):.4f} | {max(B):.4f} |")
txt = "\n".join(rows) + "\n"
print(txt)
open("experiments/memory/analysis/oscillator.md", "w").write(txt)
