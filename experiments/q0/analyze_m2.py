"""Q0 M2 : table par graine, médianes et application des critères de `analysis/m2.md`."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in glob.glob(os.path.join(HERE, "results_m2", "m2_*_it2000.json")):
    r = json.load(open(f))
    R[(r["arm"], r["opt"], r["seed"])] = r

ARMS = ("psat", "free", "pint", "ren")
for opt in ("adam", "gd"):
    seeds = sorted({s for (a, o, s) in R if o == opt})
    print(f"\n## {opt} (graines {seeds})\n")
    print("| bras | lr | " + " | ".join(f"s{s}" for s in seeds) + " | médiane | marge min (fin) | part sur ∂S ou hors de S |")
    print("|---|---|" + "---|" * len(seeds) + "---|---|---|")
    med = {}
    for a in ARMS:
        rs = [R.get((a, opt, s)) for s in seeds]
        if any(r is None for r in rs):
            continue
        t = [r["test"] for r in rs]
        med[a] = st.median(t)
        mg = "—" if a == "ren" else f"{min(r['margin_min'] for r in rs):.3g} ({st.median(r['log'][-1]['margin'] for r in rs):.3g})"
        fb = "—" if a == "ren" else f"{min(r['frac_boundary'] for r in rs):.3f}"
        print(f"| {a} | {rs[0]['lr']:g} | " + " | ".join(f"{x:.4f}" for x in t) + f" | **{med[a]:.4f}** | {mg} | {fb} |")
    if opt == "adam" and len(med) == 4:
        F, P, I, N = med["free"], med["psat"], med["pint"], med["ren"]
        bord = (F < 0.05 or I < 0.05) and P > 0.2
        pasc = F > 0.2
        print("\nCritères (Adam) :")
        print(f"- Bord confirmé (F < 0,05 ou P-int < 0,05, et P-sat > 0,2) : {bord}  [F={F:.4f}, P-int={I:.4f}, P-sat={P:.4f}]")
        print(f"- Pas la contrainte (F > 0,2) : {pasc}")
        for name, x in (("F", F), ("P-int", I)):
            print(f"- Écart P-sat → REN fermé par {name} : {(P - x) / (P - N):.0%}")
    if opt == "gd" and "ren" in med and "psat" in med:
        print(f"\nValidité GD (P-sat > 0,2 et REN < 0,05) : {med['psat'] > 0.2 and med['ren'] < 0.05}")
