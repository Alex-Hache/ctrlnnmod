"""Analyse du test d'expressivité (maître à portes). Usage : python analyze_gt.py <dossier JSON> [sortie]"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/optim/results_gated"
out = sys.argv[2] if len(sys.argv) > 2 else "experiments/optim/analysis_gated"
os.makedirs(out, exist_ok=True)
R = defaultdict(list)
for f in glob.glob(f"{src}/**/gt_*.json", recursive=True):
    z = json.load(open(f))
    R[(z["kind"], z["width"], z["seed"])].append(z)
LAB = {"ren": "REN certifié (sans portes)", "free": "mêmes équations, libres (sans portes)",
       "sren": "REN sélectif certifié (portes)", "gru": "GRU libre (portes)"}
rows = ["| Élève | largeur | params | perte finale (entr.) | perte test | pire graine (test) | graines |",
        "|---|---|---|---|---|---|---|"]
summ = {}
for w in sorted({k[1] for k in R}):
    for kind in ("ren", "free", "sren", "gru"):
        B = []
        for s in sorted({k[2] for k in R}):
            ok = [r for r in R[(kind, w, s)] if r["status"] == "ok" and r["final_train"] is not None]
            if ok:
                B.append(min(ok, key=lambda r: r["final_train"]))
        if not B:
            continue
        tr = float(np.exp(np.mean([np.log(b["final_train"]) for b in B])))
        te = float(np.exp(np.mean([np.log(b["final_test"]) for b in B])))
        worst = max(b["final_test"] for b in B)
        summ[f"{kind}/w{w}"] = dict(train=tr, test=te, worst_test=worst, nparams=B[0]["nparams"], seeds=len(B))
        rows.append(f"| {LAB[kind]} | ×{w} | {B[0]['nparams']} | {tr:.2e} | {te:.2e} | {worst:.2e} | {len(B)} |")
txt = "\n".join(rows) + "\n"
open(f"{out}/table_gt.md", "w").write(txt)
json.dump(summ, open(f"{out}/summary_gt.json", "w"), indent=1)
print(txt)
