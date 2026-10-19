r"""
Explicit discrete-time REN with a guaranteed incremental L2 gain (or contraction only).

Model (batch-first, row vectors)::

    E x+ = F x + B1 w + B2 u + bx
    Λ v  = C1 x + D12 u + bv,         w = σ(v),  σ slope-restricted in [0, 1]
    y    = C2 x + D21 w + D22 u + by

Direct parametrization (free parameters X, Y, B2, D12, C2, D21, biases)::

    H = XᵀX + εI + (KᵀK + VVᵀ)/γ,   K = [C2  D21  0],   V = [0; -D12; B2]

    H = [[H11, ·, ·], [H21, H22, ·], [H31, H32, H33]]   (blocks nx, nq, nx)
    F = H31,  B1 = H32,  P = H33,  C1 = -H21
    E = (H11 + P/ᾱ² + Y - Yᵀ)/2
    2Λᵢᵢ = Σⱼ |H22ᵢⱼ|                                    (diagonal dominance)

Guarantee (Lean: ``These.explicit_ren_dissipative``). For all parameters, with
V(Δx) = ‖EΔx‖²_{P⁻¹}::

    V(Δx+) - ᾱ² V(Δx) ≤ γ‖Δu‖² - ‖Δy‖²/γ

so the incremental L2 gain u → y is at most γ, and the model is contracting with rate ᾱ.
Without ``gamma`` (``gamma=None``): H = XᵀX + εI and only contraction holds (D22 free).

Why D11 = 0: the usual acyclic REN takes D11 = -tril(H22, -1) and solves w neuron by
neuron (nq sequential operations per time step). Here 2Λ - H22 is diagonally dominant
with a non-negative diagonal, hence positive semidefinite (thesis Lemma 3.1.1, Lean
``IsDiagDominant.posSemidef``). Replacing H22 by 2Λ keeps H positive definite, and one
step costs a few matrix products, as for a GRU. The O(n³) work (XᵀX, E⁻¹) is done once
per forward pass on a sequence, not once per step.
"""
from typing import Optional

import torch
from torch import Tensor, nn


class ExplicitREN(nn.Module):
    def __init__(self, nu: int, ny: int, nx: int, nq: int, gamma: Optional[float] = None,
                 alpha_bar: float = 1.0, eps: float = 1e-3, act: str = "tanh",
                 bias: bool = True, init_scale: float = 1.0) -> None:
        super().__init__()
        self.nu, self.ny, self.nx, self.nq = nu, ny, nx, nq
        self.gamma, self.alpha_bar, self.eps = gamma, alpha_bar, eps
        N = 2 * nx + nq
        self.X = nn.Parameter(init_scale * torch.randn(N, N) / N ** 0.5)
        self.Y = nn.Parameter(0.1 * torch.randn(nx, nx) / nx ** 0.5)
        self.B2 = nn.Parameter(torch.randn(nx, nu) / max(nu, 1) ** 0.5)
        self.D12 = nn.Parameter(torch.randn(nq, nu) / max(nu, 1) ** 0.5)
        self.C2 = nn.Parameter(torch.randn(ny, nx) / nx ** 0.5)
        self.D21 = nn.Parameter(0.1 * torch.randn(ny, nq) / nq ** 0.5)
        if gamma is None:
            self.D22 = nn.Parameter(torch.zeros(ny, nu))
        else:
            self.register_buffer("D22", torch.zeros(ny, nu))
        z = (lambda *s: nn.Parameter(torch.zeros(*s))) if bias else \
            (lambda *s: torch.zeros(*s))
        self.bx, self.bv, self.by = z(nx), z(nq), z(ny)
        self.act = {"tanh": torch.tanh, "relu": torch.relu}[act]

    # ---------------------------------------------------------------------------------
    def H(self) -> Tensor:
        nx, nq = self.nx, self.nq
        N = 2 * nx + nq
        H = self.X.T @ self.X + self.eps * torch.eye(N, dtype=self.X.dtype, device=self.X.device)
        if self.gamma is not None:
            K = torch.cat([self.C2, self.D21, torch.zeros(self.ny, nx, dtype=self.X.dtype, device=self.X.device)], 1)
            V = torch.cat([torch.zeros(nx, self.nu, dtype=self.X.dtype, device=self.X.device), -self.D12, self.B2], 0)
            H = H + (K.T @ K + V @ V.T) / self.gamma
        return H

    def frame(self) -> dict:
        nx, nq = self.nx, self.nq
        H = self.H()
        H11 = H[:nx, :nx]
        H21 = H[nx:nx + nq, :nx]
        H22 = H[nx:nx + nq, nx:nx + nq]
        H31 = H[nx + nq:, :nx]
        H32 = H[nx + nq:, nx:nx + nq]
        P = H[nx + nq:, nx + nq:]
        E = 0.5 * (H11 + P / self.alpha_bar ** 2 + self.Y - self.Y.T)
        lam = 0.5 * H22.abs().sum(1)                      # 2Λ ⪰ H22 (diag. dominance)
        Einv = torch.linalg.inv(E)
        return dict(E=E, F=H31, B1=H32, P=P, C1=-H21, Lambda=lam,
                    A=Einv @ H31, Bw=Einv @ H32, Bu=Einv @ self.B2, bx=Einv @ self.bx)

    def step(self, fr: dict, x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        v = (x @ fr["C1"].T + u @ self.D12.T + self.bv) / fr["Lambda"]
        w = self.act(v)
        y = x @ self.C2.T + w @ self.D21.T + u @ self.D22.T + self.by
        xp = x @ fr["A"].T + w @ fr["Bw"].T + u @ fr["Bu"].T + fr["bx"]
        return xp, y

    def forward(self, u: Tensor, x0: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """u: (batch, T, nu) → y: (batch, T, ny), states: (batch, T+1, nx)."""
        fr = self.frame()
        B, T, _ = u.shape
        x = torch.zeros(B, self.nx, dtype=u.dtype, device=u.device) if x0 is None else x0
        xs, ys = [x], []
        for t in range(T):
            x, y = self.step(fr, x, u[:, t])
            xs.append(x)
            ys.append(y)
        return torch.stack(ys, 1), torch.stack(xs, 1)

    # ---------------------------------------------------------------------------------
    @torch.no_grad()
    def certificate_margin(self) -> float:
        """Smallest eigenvalue of the dissipation LMI (must be > 0), in (x, w, u, x+)."""
        fr = self.frame()
        nx, nq, nu = self.nx, self.nq, self.nu
        E, F, B1, P, C1 = fr["E"], fr["F"], fr["B1"], fr["P"], fr["C1"]
        W = torch.diag(2 * fr["Lambda"])                 # D11 = 0
        g = self.gamma if self.gamma is not None else float("inf")
        Z = lambda a, b: torch.zeros(a, b, dtype=E.dtype, device=E.device)  # noqa: E731
        top = torch.cat([E + E.T - P / self.alpha_bar ** 2, -C1.T, Z(nx, nu), F.T], 1)
        mid = torch.cat([-C1, W, -self.D12, B1.T], 1)
        if g == float("inf"):
            # contraction only: drop the u row/column
            M = torch.cat([torch.cat([top[:, :nx + nq], F.T], 1),
                           torch.cat([mid[:, :nx + nq], B1.T], 1),
                           torch.cat([F, B1, P], 1)], 0)
            return float(torch.linalg.eigvalsh(0.5 * (M + M.T)).min())
        urow = torch.cat([Z(nu, nx), -self.D12.T, g * torch.eye(nu, dtype=E.dtype, device=E.device), self.B2.T], 1)
        last = torch.cat([F, B1, self.B2, P], 1)
        M = torch.cat([top, mid, urow, last], 0)
        Kfull = torch.cat([self.C2, self.D21, self.D22, Z(self.ny, nx)], 1)
        M = M - Kfull.T @ Kfull / g
        return float(torch.linalg.eigvalsh(0.5 * (M + M.T)).min())
