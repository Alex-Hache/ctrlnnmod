r"""
REN profond par couches : D₁₁ triangulaire inférieure stricte par blocs (un bloc par couche).

Les neurones de l'implicite sont groupés en L couches de tailles ``layers`` (Σ = nq) :

    D₁₁ᵢⱼ = −H₂₂ᵢⱼ   si la couche de j précède celle de i,  0 sinon
    Λᵢ    = ½ Σ_{j même couche} |H₂₂ᵢⱼ|

Un pas calcule w couche par couche (L produits matriciels). Cas limites :
``layers=[nq]`` = ExplicitREN (D₁₁ = 0) ; ``layers=[1]*nq`` = REN acyclique classique.

Garantie (Lean : ``These.layered_ren_gain``, ``These.layer_key``) : gain incrémental ≤ γ et
contraction de taux ᾱ pour tout paramètre.
"""
from typing import Optional, Sequence

import torch
from torch import Tensor

from .explicit_ren import ExplicitREN


class LayeredREN(ExplicitREN):
    def __init__(self, nu: int, ny: int, nx: int, layers: Sequence[int], gamma: Optional[float] = None,
                 alpha_bar: float = 1.0, eps: float = 1e-3, act: str = "tanh", bias: bool = True,
                 init_scale: float = 1.0) -> None:
        super().__init__(nu, ny, nx, sum(layers), gamma, alpha_bar, eps, act, bias, init_scale)
        self.layers = list(layers)
        blk = torch.cat([torch.full((s,), k) for k, s in enumerate(self.layers)])
        self.register_buffer("lower", (blk[None, :] < blk[:, None]).float())    # j avant i
        self.register_buffer("same", (blk[None, :] == blk[:, None]).float())

    def frame(self) -> dict:
        fr = super().frame()
        nx, nq = self.nx, self.nq
        H22 = self.H()[nx:nx + nq, nx:nx + nq]
        fr["D11"] = -H22 * self.lower
        fr["Lambda"] = 0.5 * (H22.abs() * self.same).sum(1)
        return fr

    def step(self, fr: dict, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        base = x @ fr["C1"].T + u @ self.D12.T + self.bv
        w = torch.zeros_like(base)
        start = 0
        for s in self.layers:
            sl = slice(start, start + s)
            v = (base[:, sl] + w @ fr["D11"][sl].T) / fr["Lambda"][sl]
            w = torch.cat([w[:, :start], self.act(v), w[:, start + s:]], 1)
            start += s
        y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T + self.by
        xp = x @ fr["A"].T + w @ fr["Bw"].T + u @ fr["Bu"].T + fr["bx"]
        return xp, y
