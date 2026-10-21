"""Résumé de M1 : tableau par classe (θ₀, θ_fin, θ*) et lecture des critères écrits dans ROADMAP.md."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_m1")
ORDER = ["blockp_nogate", "blockp_half", "blockp", "block", "blockh", "ren"]


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, int):
        return str(x)
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4):
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


rows, traj_rows, verdict = [], [], []
R = {k: json.load(open(os.path.join(RES, f"m1_{k}.json"))) for k in ORDER if os.path.exists(os.path.join(RES, f"m1_{k}.json"))}
k_ren = max(d["kappa_img"] for d in R["ren"]["traj"]) if "ren" in R else None
for k, r in R.items():
    d0, df, st = r["traj"][0], r["traj"][-1], r["star"]
    rows.append(f"| {k} | {fmt(r['test'])} | {d0['deficit']} / {df['deficit']} | {fmt(d0['n_sat'])} / {fmt(df['n_sat'])} "
                f"| {fmt(d0['kappa_img'])} / {fmt(df['kappa_img'])} | {fmt(df['frac_out'])} | {fmt(df['frac_slow'])} "
                f"| {fmt(df['jtg_ratio'])} | {fmt(df['margin'], 4)} "
                f"| {('—' if not st or 'loss' not in st else fmt(st['loss']) + ', κ=' + fmt(st['kappa_img']) + ', déf.=' + str(st['deficit']))} |")
    for d in r["traj"]:
        traj_rows.append(f"| {k} | {d['iter']} | {fmt(d['loss'])} | {d['deficit']} | {fmt(d['n_sat'])} | {fmt(d['kappa_img'])} "
                         f"| {fmt(d['sigma_min_img'])} | {fmt(d['frac_out'])} | {fmt(d['frac_slow'])} | {fmt(d['jtg_ratio'])} | {fmt(d['margin'], 4)} |")
    if k == "ren":
        continue
    rank_conf = df["deficit"] > 0 and df["frac_out"] > 0.5 and df["jtg_ratio"] < 1e-3
    rank_ref = all(d["deficit"] == 0 for d in r["traj"])
    kmax = max(d["kappa_img"] for d in r["traj"])
    metr = rank_ref and k_ren is not None and kmax >= 100 * k_ren
    v = ("rang CONFIRMÉ" if rank_conf else "rang RÉFUTÉ" if rank_ref else "rang : déficit présent, mais critère de blocage non atteint")
    v += f" ; κ_max = {fmt(kmax)} (REN : {fmt(k_ren)})"
    v += " ; métrique PLAUSIBLE" if metr else ""
    verdict.append(f"- **{k}** : {v}.")

out = ["# M1 — rang et κ de J = DΦ (blocs de 2, oscillateurs, graine 0)", "",
       "Colonnes « a / b » : θ₀ / θ_fin (2000 itér. Adam). frac_out : part de ‖g‖² hors de l'image de J. "
       "frac_slow : part de ‖g‖² sur les directions σ² < 1e-3 σ²_max. Jᵀg : ‖Jᵀg‖/(σ_max‖g‖). "
       "θ* : préimage exacte du maître (perte, κ, déficit), mesurée à 1e-6 près.", "",
       "| classe | test | déficit de rang | têtes saturées | κ(JJᵀ) image | frac_out | frac_slow | Jᵀg | marge | θ* |",
       "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
       "## Lecture des critères (écrits avant le calcul)", "", *verdict, "",
       "## Trajectoires", "",
       "| classe | itér. | perte | déficit | sat. | κ image | σ_min image | frac_out | frac_slow | Jᵀg | marge |",
       "|---|---|---|---|---|---|---|---|---|---|---|", *traj_rows, "",
       "## Contrôles", ""]
for k, r in R.items():
    c = r["checks"]
    out.append(f"- {k} : perte module {c['loss_module']:.6f}, perte en W {c['loss_W']:.6f}, "
               f"erreur J (diff. finies) {c['fd_rel_err']:.1e}" + (f" ; θ* : {r['star']['note']}" if r["star"] else ""))
print("\n".join(out))
