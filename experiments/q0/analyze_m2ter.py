"""Q0 M2-ter — applique les règles de `analysis/m2ter.md` (amendement avant calcul), sans jugement.
Écrit avant les résultats. Usage : python experiments/q0/analyze_m2ter.py [dossier]"""
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_m2ter")
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "*.json")))]
BETAS, SEEDS = (1.0, 0.5, 0.1, 0.03, 0.01), range(5)
NAME = {"A_": "Ã", "B_": "B̃", "C": "C"}


def status(r):
    """apprend (V vrai) | contact (contact avant 0,5·L0) | budget (pas de contact, perte > 0,5·L0) | div."""
    if r["status"] != "ok":
        return "div"
    if r["tc"] is None:
        return "apprend" if r["train"] <= 0.5 * r["L0"] else "budget"
    return "apprend" if r["L_pre"] <= 0.5 * r["L0"] else "contact"


def line(r):
    s = f"  s{r['seed']} "
    if r["tc"] is None:
        s += f"pas de contact en {r['iters']} pas (t·lr = {r['iters'] * r['lr']:.3g}), L/L0 = {r['train'] / r['L0']:.3f}"
    else:
        c = r["contact"]
        bl = " ".join(f"{NAME[x]} {c['blocks'][x]['real']:.3f}" for x in ("A_", "B_", "C"))
        b = c["blocks"][c["driver"]]
        s += (f"t_c = {r['tc']} (t_c·lr = {r['tc'] * r['lr']:.2e}), L(t_c−1)/L0 = {r['L_pre'] / r['L0']:.3f} | "
              f"r0 = {c['r0']:.3f}, un bloc seul : {bl} | X* = {NAME[c['driver']]}, {c['mode']} "
              f"(aléatoire méd. {b['med']:.3f}, max {b['max']:.3f}) | g_GD = {r['ggd_pre']:+.2f} | "
              f"parts Ãd/Ão/B̃C : {c['init']['Ad']:.3f}/{c['init']['Ao']:.3f}/{c['init']['BC']:.3f} → "
              f"{c['contact']['Ad']:.3f}/{c['contact']['Ao']:.3f}/{c['contact']['BC']:.3f}")
    return s + f"  ⇒ {status(r)}"


# ---------------- Étape A
A = {(r["beta"], r["seed"]): r for r in R if r["opt"] == "adam"}
print("== Étape A : P-int, Adam lr 1e-3, lr de B̃ = β·1e-3, arrêt au contact ou à 2000 pas")
nV = {}
for b in BETAS:
    rs = [A[(b, s)] for s in SEEDS if (b, s) in A]
    nV[b] = sum(status(r) == "apprend" for r in rs)
    print(f"β = {b:g} : V vrai sur {nV[b]}/{len(rs)} graines")
    for r in rs:
        print(line(r))
valid = [b for b in BETAS if nV.get(b, 0) >= 3]
bstar = max(valid) if valid else None
print(f"\nRègle : β* = {bstar if bstar is not None else 'aucun'}")
rs = [A[(0.01, s)] for s in SEEDS if (0.01, s) in A]
wc = [r for r in rs if r["tc"] is not None]
n_scale_AC = sum(r["contact"]["driver"] in ("A_", "C") and r["contact"]["mode"] == "échelle" for r in wc)
n_orient_g = sum(r["contact"]["mode"] == "orientation" and r["ggd_pre"] > 0 for r in wc)
print(f"À β = 0,01 : runs avec contact {len(wc)}/{len(rs)} ; X* ∈ {{Ã, C}} et échelle : {n_scale_AC} ; "
      f"orientation et g_GD > 0 : {n_orient_g} ; X* : {dict(Counter(NAME[r['contact']['driver']] for r in wc))}")
if bstar is not None:
    rowA = "β* existe → étape B"
elif n_scale_AC >= 3:
    rowA = "aucun β* ; X* ∈ {Ã, C} et mode échelle → même effet d'échelle d'Adam sur un autre bloc ; rien sur l'ensemble"
elif n_orient_g >= 3:
    rowA = "aucun β* ; orientation et g_GD > 0 → le gradient pousse hors de S"
else:
    rowA = "autre → ambigu"
print("Ligne de lecture A :", rowA)

# ---------------- Étape G
G = {(r["lr"], r["seed"]): r for r in R if r["opt"] == "gd"}
print("\n== Étape G : P-int, GD sans moment, clip 10, arrêt au contact ou à 5000 pas")
for eta in (0.03, 0.01):
    rs = [G[(eta, s)] for s in SEEDS if (eta, s) in G]
    if rs:
        print(f"η = {eta:g}")
        for r in rs:
            print(line(r))
flow = None
if (0.03, 0) in G and (0.01, 0) in G and G[(0.03, 0)]["tc"] is not None and G[(0.01, 0)]["tc"] is not None:
    s3, s1 = G[(0.03, 0)]["tc"] * 0.03, G[(0.01, 0)]["tc"] * 0.01
    flow = abs(s1 - s3) / s3 < 0.2
    print(f"Contrôle de régime de flot (graine 0) : t_c·η = {s3:.3g} (η 0,03) contre {s1:.3g} (η 0,01) → "
          f"{'valide' if flow else 'NON valide (écart du pas)'}")
else:
    print("Contrôle de régime de flot : non vérifiable (pas de contact pour un des deux η)")
cnt = Counter(status(G[(0.03, s)]) for s in SEEDS if (0.03, s) in G)
print("Statuts à η = 0,03 :", dict(cnt))
if cnt["contact"] >= 3:
    rowG = "contact avec V_G faux → le flot libre euclidien sort de S avant d'apprendre (T3, sans Adam)"
elif cnt["apprend"] >= 3:
    rowG = "V_G vrai → le flot euclidien apprend dans S ; contact sous Adam = métrique de l'optimiseur"
elif cnt["budget"] >= 3:
    rowG = "budget → non concluant"
else:
    rowG = "pas de majorité → non concluant"
if flow is False and rowG.startswith("contact"):
    rowG += " — MAIS régime de flot non valide : pas de lecture « flot »"
print("Ligne de lecture G :", rowG)

# ---------------- Combinaison (prévue avant calcul)
print("\n== Combinaison A + G")
if bstar is not None:
    print("β* existe : étape B d'abord, G en complément.")
elif rowG.startswith("contact") and flow is not False:
    print("Aucun β* et V_G faux : deux métriques touchent ∂S avant d'apprendre depuis W₀'. Lecture provisoire : "
          "obstacle de chemin dans S depuis cette init. Pas une preuve sur l'ensemble seul (80 % du budget dans B̃·C).")
elif rowG.startswith("V_G vrai"):
    print("Aucun β* et V_G vrai : la métrique euclidienne suffit ; l'échec sous Adam est un effet de la métrique "
          "de l'optimiseur.")
else:
    print("Pas de ligne de combinaison applicable (G non concluant ou régime de flot non valide).")
