"""Tableaux M3a : prédiction depuis les pixels et contrôleur appris en imagination.

Usage : python analyze_pix.py <dossier contenant les JSON (récursif)> [dossier de sortie]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/pixel_wm/results_modal"
out = sys.argv[2] if len(sys.argv) > 2 else "experiments/pixel_wm/analysis"
os.makedirs(out, exist_ok=True)
LAB = {"gru": "GRU (cuDNN)", "ren": "REN contractant (garanti)", "sren": "REN sélectif (garanti, Triton)",
       "mamba2": "Mamba-2"}
HOR = (10, 50, 250, 1000, 1499)

pred = defaultdict(list)
for f in glob.glob(f"{src}/**/pix_*.json", recursive=True):
    z = json.load(open(f))
    pred[(z["args"]["model"], z["args"]["n"])].append(z)
imag = defaultdict(list)
for f in glob.glob(f"{src}/**/imagine_pix_*.json", recursive=True):
    z = json.load(open(f))
    kind, n = z["wm"].split("_")[1], int(z["wm"].split("_n")[1].split("_")[0])
    imag[(kind, n)].append(z)

rows = ["| Modèle | n | graines | params | " + " | ".join(f"NMSE@{h}" for h in HOR) +
        " | min / run | contrôleur : RMSE réelle (rad) | pire run | RMSE imaginée | gradient max |",
        "|---" * (9 + len(HOR)) + "|"]
summary = {}
for kind in ("gru", "ren", "sren", "mamba2"):
    for n in (256, 1024):
        P, I = pred.get((kind, n), []), imag.get((kind, n), [])
        if not P:
            continue
        nm = {h: float(np.mean([p["eval"][f"nmse@{h}"] for p in P])) for h in HOR}
        t = float(np.mean([p["log"]["time"] for p in P])) / 60
        cells = [LAB[kind], str(n), str(len(P)), f"{P[0]['nparams'] / 1e6:.1f} M"]
        cells += [f"{nm[h]:.3f}" for h in HOR] + [f"{t:.1f}"]
        s = dict(nmse=nm, minutes=t, seeds=len(P))
        if I:
            r = [i["real_rmse_mean"] for i in I]
            g = max(max(i["log"]["grad"]) for i in I)
            cells += [f"{np.mean(r):.3f} ({len(I)})", f"{max(r):.3f}",
                      f"{np.mean([i['imagined_rmse'] for i in I]):.3f}", f"{g:.1e}"]
            s.update(real=float(np.mean(r)), worst=float(max(r)), grad_max=float(g),
                     baseline=I[0]["baseline_rmse_mean"])
        else:
            cells += ["—"] * 4
        rows.append("| " + " | ".join(cells) + " |")
        summary[f"{kind}/{n}"] = s
base = next((v["baseline"] for v in summary.values() if "baseline" in v), None)
txt = "\n".join(rows) + (f"\n\nSans politique apprise (a_t = q_ref(t+1)) : RMSE = {base:.3f} rad.\n" if base else "\n")
open(f"{out}/table_pix.md", "w").write(txt)
json.dump(summary, open(f"{out}/summary_pix.json", "w"), indent=1)
print(txt)
