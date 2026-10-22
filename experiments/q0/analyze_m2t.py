"""Q0 M2-t — applique les règles de `analysis/m2t.md`, sans jugement. Écrit avant les résultats."""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results_m2t")
RHO, SEEDS = 0.999, range(5)
maj = lambda f: sum(f) >= 3  # noqa: E731
lab = lambda x: "réussit" if x < 0.05 else "échoue" if x > 0.2 else "partiel"  # noqa: E731

print("== U1 — plateau ou lenteur")
U1 = json.load(open(os.path.join(D, "u1.json")))
for r in U1:
    print(f"  s{r['seed']}: ρ_g {r['rho_g']:.3e} | pente {r['pente']:.4f} | têtes saturées {r['n_sat']}")
if maj([r["rho_g"] < 1e-2 and r["pente"] < 0.01 for r in U1]):
    print("Ligne U1 : quasi stationnaire")
elif maj([r["pente"] > 0.05 for r in U1]):
    print("Ligne U1 : descente lente")
else:
    print("Ligne U1 : ambigu")

print("\n== U2 — barrière sur le chemin aligné")
U2 = json.load(open(os.path.join(D, "u2.json")))
ok = all(r["io_err"] < 1e-5 and abs(r["r_after"] - r["r_before"]) < 1e-5 for r in U2)
print("Contrôle u-IO (sortie et r inchangés par l'alignement) :", "passé" if ok else "ÉCHEC → pas de lecture")
bar, mono = [], []
for r in U2:
    L = r["L"]
    bar.append(max(L) > 1.1 * L[0])
    mono.append(all(L[j + 1] <= 1.01 * L[j] for j in range(len(L) - 1)))
    print(f"  s{r['seed']}: L(0) {L[0]:.4f} | max L {max(L):.4f} en λ = {L.index(max(L)) / 20:.2f} | L(1) {L[-1]:.2e} | "
          f"barrière {'oui' if bar[-1] else 'non'} | décroissant {'oui' if mono[-1] else 'non'}")
if ok:
    print("Ligne U2 :", "barrière sur ce chemin" if maj(bar) else "pas de barrière sur ce chemin (porte fermée au sens barrière : réfutée)"
          if maj(mono) else "ambigu")

print("\n== U3 / U4")
M = {}
for arm in ("pen0.03", "pen0.3", "cont", "reopen"):
    R = [json.load(open(os.path.join(D, f"{arm}_s{s}.json"))) for s in SEEDS]
    M[arm] = st.median(r["test"] for r in R)
    cf = all(r["rW_end"] <= RHO + 1e-5 for r in R)
    print(f"  {arm:8s} test : " + " ".join(f"{r['test']:.4f}" for r in R) + f" | médiane {M[arm]:.4f} ({lab(M[arm])}) | "
          f"r fin ≤ ρ : {'ok' if cf else 'ÉCHEC'} | statuts {[r['status'] for r in R]}")
best = min(M["pen0.03"], M["pen0.3"])
print("Ligne U3 :", "forcer le transfert ouvre la porte" if best < 0.05 else
      "le transfert forcé ne suffit pas" if M["pen0.03"] > 0.2 and M["pen0.3"] > 0.2 else "partiel")
if M["reopen"] < 0.05 and M["cont"] > 0.2:
    print("Ligne U4 : rouvrir suffit")
elif M["reopen"] <= M["cont"] - 0.05:
    print("Ligne U4 : aide partielle")
else:
    print("Ligne U4 : pas d'effet")
