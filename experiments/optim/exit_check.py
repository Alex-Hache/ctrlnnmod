"""Item 1b (ROADMAP) : la trajectoire libre sort-elle de l'ensemble certifié S ?

Conditions nécessaires de W ∈ S (contraction ExplicitREN, pentes du tanh dans [0, 1/λ]) :
r₀ = ρ(A) < 1 et r₁ = ρ(A + B_w Λ⁻¹ C₁) < 1. Une violation prouve la sortie de S.
Aucune violation ne prouve pas l'appartenance (test LMI exact nécessaire).

    OUT_DIR=experiments/optim/results_exit python experiments/optim/exit_check.py --run
    python experiments/optim/exit_check.py            # résumé seul
"""
import glob
import json
import multiprocessing as mp
import os
import sys

OUT = os.environ.get("OUT_DIR", "experiments/optim/results_exit")
# meilleur pas par ρ (analysis/table_ts_base.md)
BEST = {"free_gd": {0.5: 0.1, 0.9: 0.03, 0.97: 0.01, 0.995: 0.03},
        "free_adam": {0.5: 0.03, 0.9: 0.01, 0.97: 0.003, 0.995: 0.01}}
JOBS = [(rho, s, m, lr, 1500, 1, 1.0, 0) for m, d in BEST.items() for rho, lr in d.items() for s in (0, 1)]
JOBS.append((0.995, 0, "ren_adam", 0.003, 1500, 1, 1.0, 0))                  # contrôle : REN ∈ S


def summary():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/*.json")):
        r = json.load(open(f))
        lg = r["log"]
        if not lg.get("rho_A"):
            continue
        r0, r1 = max(lg["rho_A"]), max(lg["rho_A1"])
        first = next((it for it, a, b in zip(lg["it"], lg["rho_A"], lg["rho_A1"]) if a >= 1 or b >= 1), None)
        rows.append((r["method"], r["rho"], r["seed"], r["lr"], r0, r1, first, r["final_test"], r["status"]))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    print("| méthode | ρ maître | graine | pas | max r₀ | max r₁ | 1re sortie (itér.) | test final | statut |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m, rho, s, lr, r0, r1, first, te, st in rows:
        print(f"| {m} | {rho} | {s} | {lr} | {r0:.3f} | {r1:.3f} | {first if first is not None else '—'} | {te:.2e} | {st} |")


if __name__ == "__main__":
    if "--run" in sys.argv:
        sys.path.insert(0, os.path.dirname(__file__))
        os.environ["OUT_DIR"] = OUT
        os.makedirs(OUT, exist_ok=True)
        import teacher_student as ts
        ts.OUT = OUT
        with mp.get_context("fork").Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
            for line in pool.imap_unordered(ts._job, JOBS):
                print(line, flush=True)
    summary()
