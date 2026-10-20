r"""
RNN à portes par unité, certifié contractant en norme ℓ∞ pour toute suite de portes.

Modèle (lignes = lots)::

    z   = z_min + (1 − z_min) σ(W_z u + b_z)              porte par unité, pilotée par l'entrée
    f   = A x + B tanh(C x + D u + b) + E u + b_x
    x⁺  = (1 − z) ⊙ x + z ⊙ f

Certificat (Lean : ``These.gated_linf_contraction``) : si pour chaque ligne i
``Σⱼ|Aᵢⱼ| + Σₖ|Bᵢₖ| Σⱼ|Cₖⱼ| ≤ ρ < 1`` (dominance diagonale généralisée), alors pour deux
trajectoires de même entrée ``‖Δx_t‖∞ ≤ (1 − z_min(1 − ρ))ᵗ ‖Δx₀‖∞``, quelles que soient les
portes (par unité, variables dans le temps). En ℓ², c'est faux en général
(``These.l2_gate_counterexample``).

Paramétrisation directe : à partir de Ã, B̃, C libres, rᵢ = Σⱼ|Ãᵢⱼ| + Σₖ|B̃ᵢₖ| Σⱼ|Cₖⱼ| et les lignes
de [Ã | B̃] sont multipliées par ρ / (ρ + rᵢ). La somme de lignes vaut alors ρ rᵢ/(ρ + rᵢ) < ρ
pour tout paramètre.
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn


class LinfGatedRNN(nn.Module):
    def __init__(self, nu: int, nx: int, nq: Optional[int] = None, rho: float = 0.99,
                 z_min: float = 1e-3, gate_bias: float = -3.0) -> None:
        super().__init__()
        nq = nq or 2 * nx
        self.nu, self.nx, self.nq, self.rho, self.z_min = nu, nx, nq, rho, z_min
        self.A_ = nn.Parameter(torch.randn(nx, nx) / nx)
        self.B_ = nn.Parameter(torch.randn(nx, nq) / nq)
        self.C = nn.Parameter(torch.randn(nq, nx) / math.sqrt(nx))
        self.D = nn.Parameter(torch.randn(nq, nu) / math.sqrt(nu))
        self.E = nn.Parameter(torch.randn(nx, nu) / math.sqrt(nu))
        self.b, self.bx = nn.Parameter(torch.zeros(nq)), nn.Parameter(torch.zeros(nx))
        self.gate_proj = nn.Linear(nu, nx)
        nn.init.constant_(self.gate_proj.bias, gate_bias)        # portes plutôt fermées au départ

    @property
    def n_state(self) -> int:
        return self.nx

    def weights(self) -> dict:
        r = self.A_.abs().sum(1) + self.B_.abs() @ self.C.abs().sum(1)
        s = (self.rho / (self.rho + r))[:, None]
        return dict(A=self.A_ * s, B=self.B_ * s)

    def gate(self, u: Tensor) -> Tensor:
        return self.z_min + (1 - self.z_min) * torch.sigmoid(self.gate_proj(u))

    def step(self, W: dict, x: Tensor, u: Tensor, z: Optional[Tensor] = None) -> Tensor:
        z = self.gate(u) if z is None else z
        f = x @ W["A"].T + torch.tanh(x @ self.C.T + u @ self.D.T + self.b) @ W["B"].T + u @ self.E.T + self.bx
        return (1 - z) * x + z * f

    def forward(self, u: Tensor, x0: Optional[Tensor] = None) -> Tensor:
        """u : (B, T, nu) → états après chaque pas (B, T, nx)."""
        W = self.weights()
        B, T, _ = u.shape
        x = u.new_zeros(B, self.nx) if x0 is None else x0
        z = self.gate(u)                                           # toutes les portes d'un coup
        xs = []
        for t in range(T):
            x = self.step(W, x, u[:, t], z[:, t])
            xs.append(x)
        return torch.stack(xs, 1)

    @torch.no_grad()
    def certificate(self) -> dict:
        W = self.weights()
        rs = (W["A"].abs().sum(1) + W["B"].abs() @ self.C.abs().sum(1)).max().item()
        return dict(row_sum=rs, rate_bound=1 - self.z_min * (1 - rs))
