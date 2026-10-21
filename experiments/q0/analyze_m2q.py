"""Q0 M2-q — applique les règles de `analysis/m2q.md`, sans jugement. Écrit avant les résultats.
Usage : python experiments/q0/analyze_m2q.py"""
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, "results_m2q")
D3 = os.path.join(HERE, "results_m2q_T3")
RHO, SEEDS = 0.999, range(5)
med = st.median


def rho_t(arm, t):                                   # copie de m2q.rho_t (sans import de torch)
    if arm == "free":
        return math.inf
    if arm in ("p", "pgd"):
        return RHO
    if arm == "fix2":
        return 2.0
    t0, t1 = {"alent": (500, 2500), "arapide": (1500, 2000)}[arm]
    if t < t0:
        return 100.0
    if t >= t1:
        return RHO
    return 100.0 * (RHO / 100.0) ** ((t - t0) / (t1 - t0))


def majority(flags):
    return sum(flags) >= 3


# ------------------------------------------------------------------ T1
print("== T1 — par quel bloc passe la descente (GD η = 0,03, W₀', pas avant contact)")
T1 = json.load(open(os.path.join(D, "t1.json")))
a_true, a_false, b_true, b_false = [], [], [], []
for r in T1:
    S = r["steps"]
    sh = {g: st.mean(s["share"].get(g, 0.0) for s in S) for g in S[0]["share"]}
    ratio = st.mean(s["v"]["B_"] / s["v"]["A_"] for s in S)
    dp = {x: st.mean(s["dpart"][x] for s in S) for x in ("Ad", "Ao", "BC")}
    print(f"  s{r['seed']} t_c = {r['tc']} ({len(S)} pas) | parts de ‖∇L‖² : "
          + ", ".join(f"{g} {v:.3f}" for g, v in sorted(sh.items(), key=lambda kv: -kv[1]))
          + f" | v_B̃/v_Ã = {ratio:.2f} | v : Ã {st.mean(s['v']['A_'] for s in S):.3g}, B̃ {st.mean(s['v']['B_'] for s in S):.3g},"
          f" C {st.mean(s['v']['C'] for s in S):.3g} | d(part)/dt le long de −∇L : Ãd {dp['Ad']:+.3f}, Ão {dp['Ao']:+.3f}, B̃C {dp['BC']:+.3f}")
    a_true.append(sh.get("B̃", 0) > 0.5)
    a_false.append(sh.get("B̃", 0) < 0.5)
    b_true.append(ratio > 3)
    b_false.append(ratio < 1)
print("H1a (part de B̃ > 0,5) :", "VRAIE" if majority(a_true) else "FAUSSE" if majority(a_false) else "ambiguë")
print("H1b (v_B̃/v_Ã > 3) :", "VRAIE" if majority(b_true) else "FAUSSE" if majority(b_false) else "ambiguë")


# ------------------------------------------------------------------ T3
def status_g(r):
    if r["status"] != "ok":
        return "div"
    if r["tc"] is None:
        return "apprend" if r["train"] <= 0.5 * r["L0"] else "budget"
    return "apprend" if r["L_pre"] <= 0.5 * r["L0"] else "contact"


print("\n== T3 — famille β sous GD (η = 0,03, P-int, arrêt au contact ou à 5000 pas)")
R3 = [json.load(open(f)) for f in glob.glob(os.path.join(D3, "*.json"))]
G3 = {(r["beta"], r["lr"], r["seed"]): r for r in R3}
NAME = {"A_": "Ã", "B_": "B̃", "C": "C"}
cnt = {}
for b in (0.3, 0.1, 0.03):
    stt = []
    for s in SEEDS:
        r = G3.get((b, 0.03, s))
        if r is None:
            continue
        stt.append(status_g(r))
        c = r["contact"]
        extra = ("pas de contact" if r["tc"] is None else
                 f"η·t_c = {0.03 * r['tc']:.3g}, L(t_c−1)/L0 = {r['L_pre'] / r['L0']:.3f}, X* = {NAME[c['driver']]} {c['mode']}, "
                 f"parts Ãd/Ão/B̃C {c['init']['Ad']:.3f}/{c['init']['Ao']:.3f}/{c['init']['BC']:.3f} → "
                 f"{c['contact']['Ad']:.3f}/{c['contact']['Ao']:.3f}/{c['contact']['BC']:.3f}")
        print(f"  β = {b:g} s{s}: {extra} | L/L0 fin = {r['train'] / r['L0']:.3f} ⇒ {stt[-1]}")
    cnt[b] = {x: stt.count(x) for x in ("apprend", "contact", "budget", "div")}
    print(f"  β = {b:g} : {cnt[b]}")
valid = [b for b in cnt if cnt[b]["apprend"] >= 3]
bG = max(valid) if valid else None
a, f = G3.get((0.03, 0.03, 0)), G3.get((0.03, 0.01, 0))
flow = None
if a and f and a["tc"] is not None and f["tc"] is not None:
    x3, x1 = 0.03 * a["tc"], 0.01 * f["tc"]
    flow = abs(x1 - x3) / x3 < 0.2
    print(f"  régime de flot (β = 0,03, graine 0) : η·t_c = {x3:.3g} (η 0,03) contre {x1:.3g} (η 0,01) → {'valide' if flow else 'NON valide'}")
else:
    print("  régime de flot : non vérifiable (pas de contact pour un des deux η)")
if bG is not None:
    print(f"Règle T3 : β_G* = {bG:g} → sous GD, une métrique (même S) garde le flot dans S pendant qu'il apprend")
elif all(cnt[b]["contact"] >= 3 for b in cnt):
    print("Règle T3 : aucun β, contact sur la majorité → aucune métrique de la famille ne garde le flot GD dans S"
          + ("" if flow is not False else " — MAIS régime de flot non valide pour β = 0,03"))
elif any(cnt[b]["budget"] >= 3 for b in cnt):
    print("Règle T3 : « budget » sur la majorité pour au moins un β → non concluant pour ce β ; voir le détail")
else:
    print("Règle T3 : pas de ligne applicable → ambigu")

# ------------------------------------------------------------------ T4 (et T2)
print("\n== T4 / T2 — Adam lr 1e-2, 3000 itér., init W₀'")
R4 = {(r["arm"], r["seed"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(D, "t4_*.json")))}
ARMS = ("free", "p", "pgd", "fix2", "alent", "arapide")
ok_c0 = all(max(abs(R4[(a, s)]["losses"][0] - R4[("free", s)]["losses"][0]) for a in ARMS) < 1e-6 for s in SEEDS)
print("c0 (perte initiale identique pour les 6 bras) :", "passé" if ok_c0 else "ÉCHEC")
c1 = {}
for a in ("fix2", "alent", "arapide"):
    worst, nchk = 0.0, []
    for s in SEEDS:
        ra, rf = R4[(a, s)], R4[("free", s)]
        t = 0
        while t < len(ra["losses"]) and ra["rraw"][t] < rho_t(a, t):
            worst = max(worst, abs(ra["losses"][t] - rf["losses"][t]))
            t += 1
        nchk.append(t)
    c1[a] = worst < 1e-6
    print(f"c1 {a} : écart max à F' tant que r brut < ρ_t = {worst:.1e} (pas vérifiés par graine : {nchk}) → {'passé' if c1[a] else 'ÉCHEC'}")
cfin = {a: all(R4[(a, s)]["rW_end"] <= RHO + 1e-5 for s in SEEDS) for a in ("p", "pgd", "alent", "arapide")}
print("c-fin (r max de W ≤ ρ + 1e-5 à la fin) :", {a: ("passé" if v else "ÉCHEC") for a, v in cfin.items()})
M = {}
for a in ARMS:
    tests = [R4[(a, s)]["test"] for s in SEEDS]
    M[a] = med(tests)
    rr = [R4[(a, s)]["rW_end"] for s in SEEDS]
    print(f"  {a:8s} test : " + " ".join(f"{x:.4f}" for x in tests) + f" | médiane {M[a]:.4f} | r max de W en fin : "
          + " ".join(f"{x:.3g}" for x in rr) + f" | statuts : {[R4[(a, s)]['status'] for s in SEEDS]}")
for a, (t0, t1) in (("alent", (500, 2500)), ("arapide", (1500, 2000))):
    Ls = [R4[(a, s)]["losses"] for s in SEEDS]
    print(f"  {a} : perte train médiane en t = {t0} (début) {med(L[t0] for L in Ls):.4f}, en t = {t1} (fin de la décroissance) "
          f"{med(L[t1] for L in Ls):.4f}, en t = 3000 {med(L[3000] for L in Ls):.4f} ; F' en t = 3000 "
          f"{med(R4[('free', s)]['losses'][3000] for s in SEEDS):.4f}")
p_sat = [R4[("p", s)]["log"][-1]["n_sat"] for s in SEEDS]
print("  P : têtes saturées en fin :", p_sat, "| r brut max en fin :", [round(R4[("p", s)]["rraw"][-1], 3) for s in SEEDS])


def lab(x):
    return "réussit" if x < 0.05 else "échoue" if x > 0.2 else "partiel"


def gap(x):
    return (M["p"] - x) / (M["p"] - M["free"]) if M["p"] != M["free"] else float("nan")


valid4 = M["free"] < 0.05 and ok_c0
print(f"Validité T4 : F' médiane = {M['free']:.4f} → {'valide' if M['free'] < 0.05 else 'NON valide'} ; c0 {'ok' if ok_c0 else 'ÉCHEC'}")
if valid4:
    best = min(("alent", "arapide"), key=lambda a: M[a])
    A = M[best]
    note = "" if (c1[best] and cfin[best]) else " — ATTENTION : contrôle c1 ou c-fin en échec pour ce bras"
    print(f"Meilleur bras A (sélection sur 2 plannings) : {best}, médiane {A:.4f} ({lab(A)}) ; P {M['p']:.4f} ({lab(M['p'])})")
    if A < 0.05 and M["p"] > 0.2:
        print("Ligne T4 : contraindre la fin, pas le chemin, atteint un bon modèle dans S : l'obstacle est le chemin dans S, "
              "pas l'absence de bons points atteignables dans S" + note)
    elif A > 0.2 and M["p"] > 0.2:
        print("Ligne T4 : la compression perd ce que le libre a appris : finir dans S est aussi dur que rester dans S" + note)
    else:
        print(f"Ligne T4 : partiel ; part d'écart fermée g = {gap(A):.2f}" + note)
    print(f"  (autre bras A : {'arapide' if best == 'alent' else 'alent'} médiane {M['arapide' if best == 'alent' else 'alent']:.4f}, g = {gap(M['arapide' if best == 'alent' else 'alent']):.2f})")
    print(f"Ligne T2 (PGD médiane {M['pgd']:.4f}, {lab(M['pgd'])}) : ", end="")
    if M["pgd"] < 0.05 and M["p"] > 0.2:
        print("même S, issue différente : l'échec de blockp ne vient pas de S seul, mais de Φ sur ∂S. CONTRE « le bord de S est la cause »"
              + ("" if cfin["pgd"] else " — ATTENTION c-fin"))
    elif M["pgd"] > 0.2:
        print("compatible avec « S est l'obstacle » (pour cette projection)" + ("" if cfin["pgd"] else " — ATTENTION c-fin"))
    else:
        print(f"partiel ; g = {gap(M['pgd']):.2f}")
    print(f"Secondaire fix2 (ρ = 2 fixe) : médiane {M['fix2']:.4f} → {lab(M['fix2'])}")
else:
    print("T4 non lu (validité ou c0).")
print("Prédictions écrites : F' réussit, P échoue, PGD échoue, A partiel, fix2 partiel → observé :",
      {a: lab(M[a]) for a in ("free", "p", "pgd", "alent", "arapide", "fix2")})

# ------------------------------------------------------------------ T5
print("\n== T5 — contraction empirique (norme max-ℓ² par tête, 2000 pas, float64)")
p5 = os.path.join(D, "t5.json")
if os.path.exists(p5):
    R5 = {(r["arm"], r["seed"]): r for r in json.load(open(p5))}
    for a in ("p", "alent", "arapide", "pgd", "fix2", "free"):
        rows = [R5[(a, s)] for s in SEEDS if (a, s) in R5]
        print(f"  {a:8s} E : " + " ".join(f"{r['E']:.4g}" for r in rows)
              + " | d_T/d0 méd. (grande) : " + " ".join(f"{r['grande']['final_med']:.2g}" for r in rows)
              + " | (petite) : " + " ".join(f"{r['petite']['final_med']:.2g}" for r in rows))
    ctrl = all(R5[("p", s)]["E"] <= 1 + 1e-4 for s in SEEDS)
    print("Contrôle T5 (P : E ≤ 1 + 1e-4 sur les 5 graines) :", "passé" if ctrl else "ÉCHEC → pas de lecture")
    if ctrl:
        fe = [R5[("free", s)]["E"] for s in SEEDS]
        if majority([e <= 1.05 for e in fe]):
            print("Ligne T5 : F' non expansif en pratique avec r ≈", [round(R5[("free", s)]["rW_end"], 1) for s in SEEDS],
                  "→ le certificat est lâche sur la zone visitée : le bord en cause est celui du certificat")
        elif majority([e > 2 for e in fe]):
            print("Ligne T5 : F' expansif → la solution libre sort de l'ensemble non expansif (pour cette norme seulement)")
        else:
            print("Ligne T5 : ambigu")
else:
    print("  t5.json absent")
