"""
Initialisation H∞ : ancienne (« legacy », γ remplacé par le gain initial, bord de l'ensemble)
contre exacte (Lean : These.ms_variant_complete, γ prescrit gardé, marge sur M).

Système vrai : exemple (3.78) de la thèse, γ* = ‖G‖∞ = 1.
Modèle : L2BoundedLinear(param='sqrtm'), γ prescrit = 1.5 (non conservatif).

Deux scénarios d'initialisation :
- S1 « estimation sous-évaluée » : système vrai avec B × 0.5 (gain 0.5 < γ*) + bruit sur A.
- S2 « aléatoire » : système stable aléatoire de faible gain.

On suit, à chaque époque : MSE, λ_min(M), cond(P), norme du gradient, NaN éventuels.
"""
import contextlib
import io
import json
import sys

import warnings

import numpy as np
warnings.filterwarnings("ignore")
import torch

from ctrlnmod.models.ssmodels.continuous.linear.hinf import L2BoundedLinear

torch.set_default_dtype(torch.float64)

DT, NSEQ, NPTS = 0.01, 20, 500
GAMMAS = {"S1": 1.5, "S2": 1.5, "S1t": 1.05}
GAMMA = 1.5  # remplacé par scénario dans run()
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = range(5)

A_TRUE = np.array([[-2.0, 3.0], [0.0, -1.5]])
B_TRUE = np.array([[0.0], [1.0]])
C_TRUE = np.array([[1.0, 0.0]])


def hinf(A, B, C):
    w = np.concatenate([[0], np.logspace(-3, 3, 3000)])
    return max(np.linalg.norm(C @ np.linalg.solve(1j * x * np.eye(len(A)) - A, B), 2) for x in w)


def zoh(A, B):
    n, m = B.shape
    Mx = torch.zeros(n + m, n + m)
    Mx[:n, :n], Mx[:n, n:] = A, B
    E = torch.linalg.matrix_exp(Mx * DT)
    return E[:n, :n], E[:n, n:]


def simulate(A, B, C, u):
    Ad, Bd = zoh(A, B)
    x = torch.zeros(u.shape[0], A.shape[0])
    ys = []
    for k in range(u.shape[1]):
        ys.append(x @ C.T)
        x = x @ Ad.T + u[:, k] @ Bd.T
    return torch.stack(ys, 1)


def make_data(seed=1234):
    rng = np.random.default_rng(seed)
    t = np.arange(NPTS) * DT
    u = np.zeros((NSEQ, NPTS, 1))
    for s in range(NSEQ):
        f = rng.uniform(1e-3, 2, 20)
        a = rng.uniform(0.5, 1, 20)
        ph = rng.uniform(0, 2 * np.pi, 20)
        u[s, :, 0] = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t + ph[:, None])).sum(0)
    u = torch.tensor(u)
    y = simulate(*(torch.tensor(M) for M in (A_TRUE, B_TRUE, C_TRUE)), u)
    noise_std = y.std() / (10 ** (35 / 20))
    y = y + noise_std * torch.tensor(rng.normal(size=y.shape))
    return u, y


def initial_system(scenario, seed):
    rng = np.random.default_rng(seed)
    if scenario in ("S1", "S1t"):
        A = A_TRUE + 0.2 * rng.normal(size=(2, 2))
        A = A - max(0, np.max(np.linalg.eigvals(A).real) + 0.5) * np.eye(2)
        return A, 0.5 * B_TRUE, C_TRUE.copy()
    A = rng.normal(size=(2, 2))
    A = A - (np.max(np.linalg.eigvals(A).real) + 1.0) * np.eye(2)
    B, C = 0.3 * rng.normal(size=(2, 1)), 0.3 * rng.normal(size=(1, 2))
    return A, B, C


def metrics(model):
    with torch.no_grad():
        M = model.M.detach().numpy()
        P = model.P.detach().numpy()
        return float(np.linalg.eigvalsh(M).min()), float(np.linalg.cond(P))


def brl(A, B, C, P, gamma):
    nu = B.shape[1]
    top = torch.cat((A.T @ P + P @ A + C.T @ C, P @ B), 1)
    bot = torch.cat((B.T @ P, -gamma ** 2 * torch.eye(nu)), 1)
    M = torch.cat((top, bot), 0)
    return 0.5 * (M + M.T)


def run_barrier(scenario, seed, u, y, mu):
    """Paramétrisation libre (A, B, C, P) + barrière -mu logdet(-BRL). Pas sortant du domaine :
    on annule le pas et on divise le pas par 2 (backtracking)."""
    import cvxpy as cp
    A0, B0, C0 = initial_system(scenario, seed)
    nx, nu = B0.shape
    Pv, t = cp.Variable((nx, nx), symmetric=True), cp.Variable()
    Mx = cp.bmat([[A0.T @ Pv + Pv @ A0 + C0.T @ C0, Pv @ B0], [B0.T @ Pv, -GAMMA ** 2 * np.eye(nu)]])
    cp.Problem(cp.Maximize(t), [0.5 * (Mx + Mx.T) << -t * np.eye(nx + nu), Pv >> 1e-6 * np.eye(nx)]).solve()
    L0 = np.linalg.cholesky(0.5 * (Pv.value + Pv.value.T))
    params = [torch.nn.Parameter(torch.tensor(M)) for M in (A0, B0, C0, L0)]
    opt = torch.optim.Adam(params, lr=1e-2)
    hist = {"mse": [], "lam_M": [], "cond_P": [], "grad": [], "nan": False, "rejected": 0}

    def objective():
        A, B, C, L = params
        P = L @ L.T + 1e-6 * torch.eye(nx)
        mse = torch.mean((simulate(A, B, C, u) - y) ** 2)
        Lc, info = torch.linalg.cholesky_ex(-brl(A, B, C, P, GAMMA))
        if info.item() != 0:
            return mse, None, P
        return mse, mse - mu * 2 * torch.log(torch.diagonal(Lc)).sum(), P

    for ep in range(EPOCHS):
        opt.zero_grad()
        mse, loss, P = objective()
        loss.backward()
        g = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
        old = [p.detach().clone() for p in params]
        opt.step()
        for _ in range(30):
            with torch.no_grad():
                _, ok, _ = objective()
            if ok is not None:
                break
            hist["rejected"] += 1
            with torch.no_grad():
                for p, o in zip(params, old):
                    p.copy_(o + 0.5 * (p - o))
        else:
            hist["nan"] = True
            break
        with torch.no_grad():
            A, B, C, L = params
            P = L @ L.T + 1e-6 * torch.eye(nx)
            lam = torch.linalg.eigvalsh(-brl(A, B, C, P, GAMMA)).min().item()
        hist["mse"].append(mse.item())
        hist["lam_M"].append(lam)
        hist["cond_P"].append(float(np.linalg.cond(P.detach().numpy())))
        hist["grad"].append(g.item())
    A, B, C = (p.detach().numpy() for p in params[:3])
    hist["gamma_eff"] = GAMMA
    hist["final_hinf"] = hinf(A, B, C) if np.max(np.linalg.eigvals(A).real) < 0 else float("inf")
    return hist


def run(scenario, seed, method, u, y):
    global GAMMA
    GAMMA = GAMMAS[scenario]
    if method.startswith("barrier"):
        return run_barrier(scenario, seed, u, y, float(method.split("_")[1]))
    torch.manual_seed(seed)
    A0, B0, C0 = initial_system(scenario, seed)
    model = L2BoundedLinear(1, 1, 2, gamma=GAMMA, param="sqrtm")
    with contextlib.redirect_stdout(io.StringIO()):
        model.init_weights_(*(torch.tensor(M) for M in (A0, B0, C0)), GAMMA, 0.0, method=method)
    gamma_eff = float(model.gamma)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    hist = {"mse": [], "lam_M": [], "cond_P": [], "grad": [], "nan": False}
    for ep in range(EPOCHS):
        opt.zero_grad()
        A, B, C = model._frame()
        loss = torch.mean((simulate(A, B, C, u) - y) ** 2)
        if not torch.isfinite(loss):
            hist["nan"] = True
            break
        loss.backward()
        g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
        if not torch.isfinite(g):
            hist["nan"] = True
            break
        opt.step()
        lam, cond = metrics(model)
        hist["mse"].append(loss.item())
        hist["lam_M"].append(lam)
        hist["cond_P"].append(cond)
        hist["grad"].append(g.item())
    A, B, C = (M.detach().numpy() for M in model._frame())
    hist["gamma_eff"] = gamma_eff
    hist["final_hinf"] = hinf(A, B, C)
    return hist


if __name__ == "__main__":
    u, y = make_data()
    noise_floor = float(torch.mean((y - simulate(*(torch.tensor(M) for M in
                                                 (A_TRUE, B_TRUE, C_TRUE)), u)) ** 2))
    print(f"gamma* = {hinf(A_TRUE, B_TRUE, C_TRUE):.3f}, plancher de bruit MSE = {noise_floor:.2e}")
    results = {"noise_floor": noise_floor}
    for scenario in ("S1", "S2", "S1t"):
        for method in ("legacy", "exact", "barrier_1e-3", "barrier_1e-5"):
            for seed in SEEDS:
                try:
                    h = run(scenario, seed, method, u, y)
                except Exception as e:  # noqa: BLE001
                    h = {"error": repr(e)}
                results[f"{scenario}/{method}/{seed}"] = h
                if "error" in h:
                    print(scenario, method, seed, "ERREUR", h["error"][:120], flush=True)
                else:
                    print(f"{scenario} {method:12s} seed={seed} gamma_eff={h['gamma_eff']:.3f} "
                          f"mse0={h['mse'][0]:.2e} mse_fin={h['mse'][-1]:.2e} "
                          f"lamM0={h['lam_M'][0]:.1e} lamM_min={min(h['lam_M']):.1e} "
                          f"condP_max={max(h['cond_P']):.1e} nan={h['nan']} "
                          f"hinf_fin={h['final_hinf']:.3f} rej={h.get('rejected', '-')}", flush=True)
    with open("experiments/init_conditioning_results.json", "w") as f:
        json.dump(results, f)
