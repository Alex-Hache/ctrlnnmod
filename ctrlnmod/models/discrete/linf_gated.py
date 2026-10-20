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


def _row_scale(r: Tensor, rho: float, norm: str) -> Tensor:
    """Facteur par ligne qui garantit (facteur × r) ≤ ρ.
    "sat"  : ρ/(ρ + r), lisse, mais ρ n'est atteint que si r → ∞ (le bord est inatteignable) ;
    "proj" : ρ / max(r, ρ), projection radiale : identité à l'intérieur, bord atteint à r fini."""
    if norm == "proj":
        return rho / torch.clamp(r, min=rho)
    return rho / (rho + r)


class LinfGatedRNN(nn.Module):
    def __init__(self, nu: int, nx: int, nq: Optional[int] = None, rho: float = 0.99,
                 z_min: float = 1e-3, gate_bias: float = -3.0, norm: str = "sat") -> None:
        super().__init__()
        nq = nq or 2 * nx
        self.nu, self.nx, self.nq, self.rho, self.z_min, self.norm = nu, nx, nq, rho, z_min, norm
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
        s = _row_scale(r, self.rho, self.norm)[:, None]
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


class BlockGatedRNN(nn.Module):
    r"""Généralisation par blocs : têtes de taille k, une porte par tête, norme mixte
    ``‖x‖ = maxₕ ‖xₕ‖₂``.

    Certificat : pour chaque tête h, ``Σₖ‖A_{hk}‖₂ + Σ_q ‖B_{h,q}‖₂ Σₖ‖C_{q,k}‖₂ ≤ ρ < 1`` (normes
    spectrales des blocs). Alors ``‖Δxₕ⁺‖₂ ≤ (1 − zₕ(1 − ρ)) maxₖ‖Δxₖ‖₂`` pour toutes portes
    ``zₕ ∈ [0, 1]``. Avec k = 1, on retrouve ``LinfGatedRNN`` (portes par unité). Avec k = 2, un
    bloc peut être une rotation amortie ``r R(θ)`` (oscillateur), ce que la classe ℓ∞ exclut : une
    rotation de 45° a une somme de lignes √2 r.
    """

    def __init__(self, nu: int, n_heads: int, k: int = 2, nq: Optional[int] = None, rho: float = 0.99,
                 z_min: float = 1e-3, gate_bias: float = -3.0, norm: str = "sat",
                 ungated_heads: int = 0) -> None:
        super().__init__()
        nx = n_heads * k
        self.ungated = ungated_heads              # têtes sans porte (z ≡ 1) : dynamique rapide libre
        nq = nq or 2 * nx
        self.nu, self.H, self.k, self.nx, self.nq, self.rho, self.z_min = nu, n_heads, k, nx, nq, rho, z_min
        self.norm = norm
        self.A_ = nn.Parameter(torch.randn(nx, nx) / nx)
        self.B_ = nn.Parameter(torch.randn(nx, nq) / nq)
        self.C = nn.Parameter(torch.randn(nq, nx) / math.sqrt(nx))
        self.D = nn.Parameter(torch.randn(nq, nu) / math.sqrt(nu))
        self.E = nn.Parameter(torch.randn(nx, nu) / math.sqrt(nu))
        self.b, self.bx = nn.Parameter(torch.zeros(nq)), nn.Parameter(torch.zeros(nx))
        self.gate_proj = nn.Linear(nu, n_heads)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

    @property
    def n_state(self) -> int:
        return self.nx

    def _block_rows(self) -> Tensor:
        H, k = self.H, self.k
        Ab = self.A_.view(H, k, H, k).permute(0, 2, 1, 3)                     # (H, H, k, k)
        nA = torch.linalg.matrix_norm(Ab, ord=2) if k > 1 else Ab.abs()[..., 0, 0]
        nB = self.B_.view(H, k, self.nq).norm(dim=1)                          # (H, nq)
        c = self.C.view(self.nq, H, k).norm(dim=2).sum(1)                     # (nq,)
        return nA.sum(1) + nB @ c                                             # (H,)

    def weights(self) -> dict:
        s = _row_scale(self._block_rows(), self.rho, self.norm).repeat_interleave(self.k)[:, None]
        return dict(A=self.A_ * s, B=self.B_ * s)

    def gate(self, u: Tensor) -> Tensor:
        z = self.z_min + (1 - self.z_min) * torch.sigmoid(self.gate_proj(u))
        if self.ungated:
            z = torch.cat([torch.ones_like(z[..., :self.ungated]), z[..., self.ungated:]], -1)
        return z.repeat_interleave(self.k, dim=-1)

    def step(self, W: dict, x: Tensor, u: Tensor, z: Optional[Tensor] = None) -> Tensor:
        z = self.gate(u) if z is None else z
        f = x @ W["A"].T + torch.tanh(x @ self.C.T + u @ self.D.T + self.b) @ W["B"].T + u @ self.E.T + self.bx
        return (1 - z) * x + z * f

    def forward(self, u: Tensor, x0: Optional[Tensor] = None) -> Tensor:
        W = self.weights()
        B, T, _ = u.shape
        x = u.new_zeros(B, self.nx) if x0 is None else x0
        z = self.gate(u)
        xs = []
        for t in range(T):
            x = self.step(W, x, u[:, t], z[:, t])
            xs.append(x)
        return torch.stack(xs, 1)

    @torch.no_grad()
    def certificate(self) -> dict:
        r = self._block_rows()
        rs = (_row_scale(r, self.rho, self.norm) * r).max().item()
        return dict(row_sum=rs, rate_bound=1 - self.z_min * (1 - rs))
