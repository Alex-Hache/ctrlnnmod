"""
Expérience 1 — boucle fermée : améliorer un contrôleur de bras robotique sans jamais
perdre la stabilité.

Système : bras plan 2 axes dans le plan vertical (gravité), couple aux articulations.
Contrôleur de base : PD + compensation de gravité (globalement asymptotiquement stable).
Tâche : aller de q0 à q* en contournant un obstacle circulaire (le PD seul le traverse).

Politiques apprises (couple résiduel ajouté au contrôleur de base) :
- "mlp"     : retour d'état libre  u = K(x) + π_θ(x)                 (choix RL par défaut)
- "pb_rnn"  : u = K(x) + M_θ(ŵ), M_θ = RNN tanh libre                (structure sans garantie)
- "pb_ren"  : u = K(x) + M_θ(ŵ), M_θ = ExplicitREN de gain ≤ γ       (garantie, ctrlnmod)
ŵ_t = x_t − f̂(x_{t−1}, u_{t−1}) est la perturbation reconstruite avec le modèle nominal ;
ŵ_0 = x_0 − x*. Si le modèle est exact, ŵ = perturbation réelle : M_θ(ŵ) ∈ ℓ2 et la boucle
reste stable pour TOUT θ (performance boosting, Furieri et al.).
"""
import math

import torch
from torch import nn

from ctrlnmod.models.discrete.explicit_ren import ExplicitREN

DT = 0.02
G = 9.81
NOMINAL = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, damp=0.5)
KP, KD = 40.0, 12.0
Q0 = torch.tensor([-2.6, 0.6])        # bras vers le bas-gauche
QSTAR = torch.tensor([-0.55, -0.6])   # bras vers le bas-droite
OBST_C = torch.tensor([0.0, -1.95])   # obstacle sur le point bas de la trajectoire de l'effecteur
OBST_R = 0.18


# ----------------------------------------------------------------------------------------
# Dynamique
# ----------------------------------------------------------------------------------------
def gravity(q, p):
    q1, q12 = q[..., 0], q[..., 0] + q[..., 1]
    g1 = (p["m1"] + p["m2"]) * G * p["l1"] * torch.cos(q1) + p["m2"] * G * p["l2"] * torch.cos(q12)
    g2 = p["m2"] * G * p["l2"] * torch.cos(q12)
    return torch.stack([g1, g2], -1)


def accel(x, tau, p):
    q, dq = x[..., :2], x[..., 2:]
    m1, m2, l1, l2 = p["m1"], p["m2"], p["l1"], p["l2"]
    c2, s2 = torch.cos(q[..., 1]), torch.sin(q[..., 1])
    M11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2
    M12 = m2 * l2 ** 2 + m2 * l1 * l2 * c2
    M22 = m2 * l2 ** 2 * torch.ones_like(c2)
    h = m2 * l1 * l2 * s2
    cor = torch.stack([-h * (2 * dq[..., 0] * dq[..., 1] + dq[..., 1] ** 2), h * dq[..., 0] ** 2], -1)
    rhs = tau - cor - gravity(q, p) - p["damp"] * dq
    det = M11 * M22 - M12 ** 2
    a1 = (M22 * rhs[..., 0] - M12 * rhs[..., 1]) / det
    a2 = (-M12 * rhs[..., 0] + M11 * rhs[..., 1]) / det
    return torch.stack([a1, a2], -1)


def f(x, tau, p):
    """Euler semi-implicite."""
    dq = x[..., 2:] + DT * accel(x, tau, p)
    q = x[..., :2] + DT * dq
    return torch.cat([q, dq], -1)


def base_controller(x):
    """PD + compensation de gravité avec le modèle NOMINAL."""
    q, dq = x[..., :2], x[..., 2:]
    return gravity(q, NOMINAL) + KP * (QSTAR - q) - KD * dq


def link_points(q, p=NOMINAL, n=6):
    """Points le long des deux segments (pour la collision)."""
    q1, q12 = q[..., :1], q[..., :1] + q[..., 1:2]
    s = torch.linspace(0, 1, n)
    elbow = torch.cat([p["l1"] * torch.cos(q1), p["l1"] * torch.sin(q1)], -1)
    pts1 = s[:, None] * elbow[..., None, :]
    ee_dir = torch.cat([torch.cos(q12), torch.sin(q12)], -1)
    pts2 = elbow[..., None, :] + p["l2"] * s[:, None] * ee_dir[..., None, :]
    return torch.cat([pts1, pts2], -2)


def point_dists(q):
    return (link_points(q) - OBST_C).norm(dim=-1) - OBST_R


def obstacle_dist(q):
    return point_dists(q).min(-1).values


# ----------------------------------------------------------------------------------------
# Politiques
# ----------------------------------------------------------------------------------------
class MLPPolicy(nn.Module):
    kind = "mlp"

    def __init__(self, h=64, scale=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 2))
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    m.weight.mul_(scale)
            self.net[-1].weight.mul_(10.0)   # couples en N·m

    def reset(self, B):
        pass

    def __call__(self, x, w_hat):
        return self.net(x - torch.cat([QSTAR, torch.zeros(2)]))


class PBRNN(nn.Module):
    """Performance boosting avec un RNN tanh libre (aucune garantie de stabilité)."""
    kind = "pb_rnn"

    def __init__(self, nx=32, scale=1.0):
        super().__init__()
        self.nx = nx
        self.W = nn.Parameter(scale * torch.randn(nx, nx) / nx ** 0.5)
        self.U = nn.Parameter(scale * torch.randn(nx, 4) / 2)
        self.C = nn.Parameter(10.0 * scale * torch.randn(2, nx) / nx ** 0.5)
        self.b = nn.Parameter(torch.zeros(nx))

    def reset(self, B):
        self.h = torch.zeros(B, self.nx)

    def __call__(self, x, w_hat):
        self.h = torch.tanh(self.h @ self.W.T + w_hat @ self.U.T + self.b)
        return self.h @ self.C.T


class PBREN(nn.Module):
    """Performance boosting avec un REN explicite de gain incrémental ≤ γ (garanti)."""
    kind = "pb_ren"

    def __init__(self, nx=16, nq=32, gamma=200.0, scale=1.0, out_scale=10.0):
        super().__init__()
        # gain garanti de ŵ vers le couple : out_scale · (gamma / out_scale) = gamma
        self.out_scale = out_scale
        self.ren = ExplicitREN(4, 2, nx, nq, gamma=gamma / out_scale, init_scale=scale, bias=False)
        with torch.no_grad():
            for p in self.ren.parameters():
                p.mul_(scale)

    def reset(self, B):
        self.fr = self.ren.frame()
        self.h = torch.zeros(B, self.ren.nx)

    def __call__(self, x, w_hat):
        self.h, y = self.ren.step(self.fr, self.h, w_hat)
        return self.out_scale * y


# ----------------------------------------------------------------------------------------
# Simulation en boucle fermée
# ----------------------------------------------------------------------------------------
def rollout(policy, x0, T, plant=NOMINAL, pushes=None):
    """pushes : tenseur (B, T, 4) de perturbations additives sur l'état, ou None."""
    B = x0.shape[0]
    policy.reset(B)
    x = x0
    xstar = torch.cat([QSTAR, torch.zeros(2)])
    w_hat = x0 - xstar
    xs, us = [x], []
    for t in range(T):
        u_res = policy(x, w_hat)
        u = base_controller(x) + u_res
        x_next = f(x, u, plant)
        if pushes is not None:
            x_next = x_next + pushes[:, t]
        w_hat = x_next - f(x, u, NOMINAL)       # perturbation reconstruite (modèle nominal)
        x = x_next
        xs.append(x)
        us.append(u_res)
    return torch.stack(xs, 1), torch.stack(us, 1)


def cost(xs, us):
    q, dq = xs[..., :2], xs[..., 2:]
    T = xs.shape[1]
    wt = torch.linspace(0.2, 1.0, T) ** 2
    track = (wt * ((q - QSTAR) ** 2).sum(-1)).mean()
    vel = 0.01 * (dq ** 2).sum(-1).mean()
    d = point_dists(q)                       # pénalité sur chaque point des segments
    coll = 50.0 * (torch.relu(0.2 - d) ** 2).sum(-1).mean() + 2000.0 * (torch.relu(-d) ** 2).sum(-1).mean()
    effort = 1e-4 * (us ** 2).sum(-1).mean()
    return track + vel + coll + effort


def sample_x0(B, spread=0.15, gen=None):
    q = Q0 + spread * torch.randn(B, 2, generator=gen)
    dq = 0.3 * spread * torch.randn(B, 2, generator=gen)
    return torch.cat([q, dq], -1)


@torch.no_grad()
def evaluate(policy, x0, T, plant=NOMINAL, pushes=None):
    xs, us = rollout(policy, x0, T, plant, pushes)
    xstar = torch.cat([QSTAR, torch.zeros(2)])
    finite = torch.isfinite(xs).all(-1).all(-1)
    err_end = torch.where(finite, (xs[:, -1] - xstar).norm(dim=-1), torch.full((x0.shape[0],), math.inf))
    diverged = (~finite) | (xs.abs().amax((1, 2)) > 1e3)
    dmin = torch.where(finite, obstacle_dist(xs[..., :2]).amin(1), torch.full((x0.shape[0],), -math.inf))
    collided = dmin < 0
    reached = err_end < 0.05
    return dict(reached=reached.float().mean().item(), collided=collided.float().mean().item(),
                diverged=diverged.float().mean().item(), success=(reached & ~collided).float().mean().item(),
                err_end_median=err_end.median().item(), max_tau=us.abs().amax().item() if finite.all() else math.inf)
