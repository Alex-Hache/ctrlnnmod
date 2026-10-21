"""Q0 M2-q T7 — observabilité : grammiens empiriques, corrélation têtes, élagage têtes et unités tanh.
Critères : `analysis/m2q.md` § T7 (a0c910f). Règles codées ici, écrites avant les résultats.
Usage : PYTHONPATH=. python experiments/q0/m2q_obs.py"""
import copy
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_boundary as m2  # noqa: E402
import m2bis as mb  # noqa: E402
import m2q  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
RHO, D = m2.RHO, os.path.join(HERE, "results_m2q")
AMPS, CLASSES = (1e-3, 1e-2, 1e-1, 1.0), ("nominale", "nulle")


def f32(fn, *a):
    """Génère avec le type par défaut float32 (le tirage aléatoire dépend du type), puis convertit en float64."""
    torch.set_default_dtype(torch.float32)
    try:
        out = fn(*a)
    finally:
        torch.set_default_dtype(torch.float64)
    return [t.double() for t in out] if isinstance(out, (tuple, list)) else out.double()


def load(arm, seed):
    model = m2q.build4(arm, seed)
    model.load_state_dict(torch.load(os.path.join(D, f"t4_{arm}_s{seed}.pt")))
    model = model.double()
    if arm == "fix2":
        model.m.rho = 2.0
    return model


# ------------------------------------------------------------------ grammiens empiriques (générique)
def gramians(sim, n, nu, u, c):
    """sim(u, x0) -> (états (N, T, n), sorties (N, T, ny)). Différences centrées."""
    N = u.shape[0]
    Wo = torch.zeros(n, n)
    dy = []
    for i in range(n):
        e = torch.zeros(N, n)
        e[:, i] = c
        dy.append(sim(u, e)[1] - sim(u, -e)[1])
    dy = torch.stack(dy)                                               # (n, N, T, ny)
    Wo = torch.einsum("inty,jnty->ij", dy, dy) / (4 * c * c * N)
    Wc = torch.zeros(n, n)
    for mi in range(nu):
        up, um = u.clone(), u.clone()
        up[:, 0, mi] += c
        um[:, 0, mi] -= c
        dx = sim(up, torch.zeros(N, n))[0] - sim(um, torch.zeros(N, n))[0]
        Wc += torch.einsum("nti,ntj->ij", dx, dx) / (4 * c * c * N)
    return Wo, Wc


def hankel(Wo, Wc):
    ev = torch.linalg.eigvals(Wc @ Wo).real.clamp(min=0)
    return torch.sort(torch.sqrt(ev), descending=True).values


def model_sim(model):
    def sim(u, x0):
        xs = model.m.forward(u, x0)
        return xs, model.C(xs)
    return sim


# ------------------------------------------------------------------ contrôle g0 : maître linéaire
def check_g0(seed=0):
    A, B, C = f32(m2.osc.teacher, seed)
    T = 300

    def sim(u, x0):
        x, xs = x0, []
        for t in range(u.shape[1]):
            x = x @ A.T + u[:, t] @ B.T
            xs.append(x)
        xs = torch.stack(xs, 1)
        return xs, xs @ C.T
    u = f32(m2.osc.inputs, 32, T, 40 + seed)
    Wo_an, P = torch.zeros(8, 8), torch.eye(8)
    for t in range(1, T + 1):
        P = A @ P
        Wo_an += P.T @ C.T @ C @ P
    out = {}
    for c in AMPS:
        Wo, _ = gramians(sim, 8, 2, u, c)
        out[c] = float((Wo - Wo_an).norm() / Wo_an.norm())
    return out


# ------------------------------------------------------------------ r et élagage
def r_max(m, heads):
    with torch.no_grad():
        r = mb.r_raw(m.H, m.k, m.nq, m.A_, m.B_, m.C)
    return float(r[heads].max()) if len(heads) else 0.0


@torch.no_grad()
def drop_head(model, h):
    m, k = model.m, model.m.k
    s = slice(h * k, (h + 1) * k)
    m.A_[s, :] = 0
    m.A_[:, s] = 0
    m.B_[s, :] = 0
    m.E[s, :] = 0
    m.bx[s] = 0
    m.C[:, s] = 0
    model.C.weight[:, s] = 0


@torch.no_grad()
def drop_unit(model, q):
    m = model.m
    m.B_[:, q] = 0
    m.C[q, :] = 0
    m.D[q, :] = 0
    m.b[q] = 0


def mse(model, u, y):
    with torch.no_grad():
        return float(((model(u) - y) ** 2).mean())


def prune(model, kind, u_tr, y_tr, u_te, y_te):
    n = model.m.H if kind == "tête" else model.m.nq
    drop = drop_head if kind == "tête" else drop_unit
    cur, kept, path = copy.deepcopy(model), list(range(n)), []
    heads = lambda keptset: (keptset if kind == "tête" else list(range(model.m.H)))   # noqa: E731
    path.append(dict(k=0, test=mse(cur, u_te, y_te), r=r_max(cur.m, heads(kept))))
    for k in range(1, n):
        best = None
        for j in kept:
            cand = copy.deepcopy(cur)
            drop(cand, j)
            L = mse(cand, u_tr, y_tr)
            if best is None or L < best[0]:
                best = (L, j, cand)
        _, j, cur = best
        kept.remove(j)
        path.append(dict(k=k, removed=j, test=mse(cur, u_te, y_te), r=r_max(cur.m, heads(kept))))
    return path


def spearman(a, b):
    ra = torch.argsort(torch.argsort(torch.tensor(a))).double()
    rb = torch.argsort(torch.argsort(torch.tensor(b))).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


# ------------------------------------------------------------------ run
lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


g0 = check_g0()
ok_g0 = all(v < 1e-6 for v in g0.values())
say("g0 (maître linéaire, écart relatif W_o empirique / analytique) : "
    + ", ".join(f"c={c:g} : {v:.1e}" for c, v in g0.items()) + f" → {'passé' if ok_g0 else 'ÉCHEC'}")

R = {}
for arm in ("free", "p", "fix2"):
    for seed in range(5):
        model = load(arm, seed)
        m = model.m
        with torch.no_grad():
            rh = mb.r_raw(m.H, m.k, m.nq, m.A_, m.B_, m.C) if arm == "free" else m2.rows(m, *m.weights().values())
        rh = [float(x) for x in rh]
        rec = dict(arm=arm, seed=seed, r_h=rh, sweep={})
        for cls in CLASSES:
            u = f32(m2.osc.inputs, 32, 300, 40 + seed)
            if cls == "nulle":
                u = torch.zeros_like(u)
            for c in AMPS:
                Wo, Wc = gramians(model_sim(model), m.nx, 2, u, c)
                hs = hankel(Wo, Wc)
                oh = [float(torch.trace(Wo[2 * h:2 * h + 2, 2 * h:2 * h + 2])) for h in range(m.H)]
                rec["sweep"][f"{cls}|{c:g}"] = dict(
                    n_eff=int((hs > 1e-3 * hs[0]).sum()), hankel=[float(x / hs[0]) for x in hs],
                    o_h=oh, spearman=spearman(oh, rh))
        R[(arm, seed)] = rec
        main = rec["sweep"]["nominale|0.01"]
        say(f"{arm:5s} s{seed} : r_h max {max(rh):.3g} | Spearman(o_h, r_h) à c=1e-2 nominale : {main['spearman']:+.2f} "
            f"(balayage : " + " ".join(f"{v['spearman']:+.2f}" for v in rec["sweep"].values()) + ") | n_eff (c=1e-2 nom.) "
            f"{main['n_eff']} | Hankel σ_i/σ₁ : " + " ".join(f"{x:.1e}" for x in main["hankel"][:12]))

say("\n== Élagage (libre)")
P = {}
for seed in range(5):
    model = load("free", seed)
    u_tr, y_tr, u_te, y_te = f32(m2.data, seed)
    t4 = json.load(open(os.path.join(D, f"t4_free_s{seed}.json")))["test"]
    P[seed] = {kind: prune(model, kind, u_tr, y_tr, u_te, y_te) for kind in ("tête", "unité")}
    g1 = abs(P[seed]["tête"][0]["test"] - t4)
    for kind in ("tête", "unité"):
        say(f"s{seed} {kind:5s} : " + " | ".join(f"k={e['k']} test {e['test']:.3g} r {e['r']:.3g}"
                                                for e in P[seed][kind] if kind == "tête" or e["k"] % 4 == 0 or e["k"] == 31))
    P[seed]["g1"] = g1

ok_g1 = all(P[s]["g1"] < 1e-9 for s in range(5))
say(f"\ng1 (k = 0 reproduit le test de T4, écart < 1e-9) : {'passé' if ok_g1 else 'ÉCHEC'} "
    f"(écarts : {[f'{P[s][chr(103)+chr(49)]:.1e}' for s in range(5)]})")

# ------------------------------------------------------------------ règles
if not (ok_g0 and ok_g1):
    say("Contrôle en échec → pas de lecture.")
else:
    sp = [R[("free", s)]["sweep"]["nominale|0.01"]["spearman"] for s in range(5)]
    H7a = "SOUTIENT" if sum(x < -0.5 for x in sp) >= 3 else "CONTRE" if sum(x > 0 for x in sp) >= 3 else "ambiguë"
    say(f"H7a (Spearman(o_h, r_h) < −0,5 sur la majorité) : {H7a} ; valeurs : " + " ".join(f"{x:+.2f}" for x in sp))
    fs, inS = [], []
    for s in range(5):
        r_full = P[s]["tête"][0]["r"]
        cands = [e["r"] for kind in ("tête", "unité") for e in P[s][kind] if e["test"] <= 0.05 and e["r"] > 0]
        rmin = min(cands) if cands else r_full
        fs.append(r_full / rmin)
        inS.append(rmin <= RHO)
        say(f"  s{s} : r complet {r_full:.3g}, plus petit r gardé avec test ≤ 0,05 : {rmin:.3g} → f = {r_full / rmin:.2f}")
    if sum(f >= 10 for f in fs) >= 3:
        H7b = "SOUTIENT : l'essentiel de r est dans des directions dont la sortie n'a pas besoin" + \
              (" ; le modèle utile est dans S" if sum(inS) >= 3 else "")
    elif sum(f < 2 for f in fs) >= 3:
        H7b = "RÉFUTE (sans réentraînement) : r est porté par ce dont la sortie a besoin"
    else:
        H7b = "partiel"
    say(f"H7b : {H7b} ; f = " + " ".join(f"{f:.2f}" for f in fs))

json.dump(dict(g0=g0, R={f"{a}_{s}": v for (a, s), v in R.items()},
               prune={str(s): {k: v for k, v in P[s].items()} for s in P}), open(os.path.join(D, "t7.json"), "w"))
open(os.path.join(HERE, "m2q_T7_analysis.txt"), "w").write("\n".join(lines) + "\n")
