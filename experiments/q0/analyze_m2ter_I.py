"""Q0 M2-ter, étape I — applique les règles de `analysis/m2ter.md` (étape I), sans jugement.
Écrit avant les résultats. Usage : python experiments/q0/analyze_m2ter_I.py [dossier]"""
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_m2ter_I")
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "*.json")))]
S = {(r["init"], r["opt"], r["lr"], r["seed"]): r for r in R}
NAME = {"A_": "Ã", "B_": "B̃", "C": "C"}


def status(r):
    if r["status"] != "ok":
        return "div"
    if r["tc"] is None:
        return "apprend" if r["train"] <= 0.5 * r["L0"] else "budget"
    return "apprend" if r["L_pre"] <= 0.5 * r["L0"] else "contact"


def line(r):
    s = f"  s{r['seed']} L0 = {r['L0']:.3f} "
    if r["tc"] is None:
        s += f"pas de contact en {r['iters']} pas (t·lr = {r['iters'] * r['lr']:.3g}), L/L0 = {r['train'] / r['L0']:.3f}"
    else:
        c = r["contact"]
        b = c["blocks"][c["driver"]]
        bl = " ".join(f"{NAME[x]} {c['blocks'][x]['real']:.3f}" for x in ("A_", "B_", "C"))
        s += (f"t_c = {r['tc']} (t_c·lr = {r['tc'] * r['lr']:.3g}), L(t_c−1)/L0 = {r['L_pre'] / r['L0']:.3f} | "
              f"un bloc seul : {bl} | X* = {NAME[c['driver']]}, {c['mode']} (aléatoire méd. {b['med']:.3f}, "
              f"max {b['max']:.3f}) | parts Ãd/Ão/B̃C : {c['init']['Ad']:.3f}/{c['init']['Ao']:.3f}/"
              f"{c['init']['BC']:.3f} → {c['contact']['Ad']:.3f}/{c['contact']['Ao']:.3f}/{c['contact']['BC']:.3f}")
    return s + f"  ⇒ {status(r)}"


def majority(init, opt, lr):
    c = Counter(status(S[(init, opt, lr, s)]) for s in range(5) if (init, opt, lr, s) in S)
    for lab in ("contact", "apprend", "budget"):
        if c[lab] >= 3:
            return lab, dict(c)
    return "pas de majorité", dict(c)


maj, flow = {}, {}
for init in ("i1", "i2"):
    for opt, lr in (("gd", 0.03), ("adam", 0.001)):
        print(f"== {init.upper()} — {'GD η = 0,03' if opt == 'gd' else 'Adam lr 1e-3, β = 1'}")
        for s in range(5):
            if (init, opt, lr, s) in S:
                print(line(S[(init, opt, lr, s)]))
        maj[(init, opt)] = majority(init, opt, lr)
        print("   statuts :", maj[(init, opt)][1], "→ majorité :", maj[(init, opt)][0])
    a, b = S.get((init, "gd", 0.03, 0)), S.get((init, "gd", 0.01, 0))
    if a and b and a["tc"] is not None and b["tc"] is not None:
        x3, x1 = a["tc"] * 0.03, b["tc"] * 0.01
        flow[init] = abs(x1 - x3) / x3 < 0.2
        print(f"   régime de flot (graine 0) : η·t_c = {x3:.3g} (η 0,03) contre {x1:.3g} (η 0,01) → "
              f"{'valide' if flow[init] else 'NON valide'}")
    else:
        flow[init] = None
        print("   régime de flot : non vérifiable (pas de contact pour un des deux η)")
    print()

g1, g2 = maj[("i1", "gd")][0], maj[("i2", "gd")][0]
rows = {("contact", "contact"): "le flot euclidien sort de S avant d'apprendre, même avec marge 0,5 et budget dans Ã : "
                                "l'obstacle ne vient pas de W₀'. Indice pour l'ensemble.",
        ("contact", "apprend"): "la répartition du budget à l'init décide : obstacle lié à l'init, pas à S.",
        ("apprend", "apprend"): "la marge suffit : W₀' était trop près de ∂S.",
        ("apprend", "contact"): "inattendu ; ambigu."}
print(f"== Lecture GD : I1 = {g1}, I2 = {g2}")
print("Ligne :", rows.get((g1, g2), "non concluant (budget ou pas de majorité)"))
bad = [i for i in ("i1", "i2") if flow[i] is False]
if bad:
    print("ATTENTION : régime de flot non valide pour", bad, "→ pas de lecture « flot » pour ces inits.")
print("\n== Prédiction Adam (contact pour I1 et I2) :",
      "confirmée" if maj[("i1", "adam")][0] == maj[("i2", "adam")][0] == "contact"
      else f"NON confirmée (I1 : {maj[('i1', 'adam')][0]}, I2 : {maj[('i2', 'adam')][0]})")
print("== Prédiction GD (contact pour I1 et I2) :", "confirmée" if (g1, g2) == ("contact", "contact")
      else f"NON confirmée (I1 : {g1}, I2 : {g2})")
