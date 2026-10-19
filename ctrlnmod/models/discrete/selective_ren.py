r"""
REN sélectif : couche récurrente de type Mamba-2 avec un cœur REN certifié.

Pour chaque tête h (H têtes, état x_h ∈ R^nx) :

    f_h(x, u) = ExplicitREN_h(x, u)                       (cœur certifié, même paramétrisation)
    g_h(u)    = g_min + (1 − g_min) σ(W_g u + b_g)_h        (porte scalaire, choisie par l'entrée)
    x_h⁺      = (1 − g_h) x_h + g_h f_h(x_h, u)

Comme dans Mamba-2, la porte est un scalaire par tête : g petit garde l'état (mémoire), g grand
le remplace par la réponse du cœur. La porte peut dépendre de l'entrée de façon quelconque.

Garantie (Lean : ``These.selective_ren_contraction``). Pour toute suite d'entrées et toute suite
de portes dans [g_min, 1], deux trajectoires de même entrée vérifient, tête par tête,

    V(Δx_t) ≤ (1 − g_min (1 − ᾱ²))^t · V(Δx_0),     V(z) = (Ez)ᵀ P⁻¹ (Ez).

Preuve : V est une forme quadratique positive donc convexe ; le cœur contracte V d'un facteur
ᾱ² (``explicit_ren_step`` avec Δu = 0). La porte étant scalaire, la métrique peut être quelconque.

Si les portes ne dépendent pas de l'entrée perturbée (canal de sélection séparé), le gain
incrémental pondéré vérifie de plus Σ g‖Δy‖² ≤ γ² Σ g‖Δu‖² + γV₀ (``selective_ren_gain``).
"""
import torch
from torch import Tensor, nn


class SelectiveREN(nn.Module):
    def __init__(self, nu: int, n_heads: int = 8, nx: int = 16, nq: int = 32, gamma: float = 1.0,
                 alpha_bar: float = 0.9, eps: float = 1e-3, g_min: float = 0.02, act: str = "tanh") -> None:
        super().__init__()
        H, N = n_heads, 2 * nx + nq
        self.nu, self.H, self.nx, self.nq, self.N = nu, H, nx, nq, N
        self.gamma, self.alpha_bar, self.eps, self.g_min = gamma, alpha_bar, eps, g_min
        self.X = nn.Parameter(torch.randn(H, N, N) / N ** 0.5)
        self.Y = nn.Parameter(0.1 * torch.randn(H, nx, nx) / nx ** 0.5)
        self.B2 = nn.Parameter(torch.randn(H, nx, nu) / nu ** 0.5)
        self.D12 = nn.Parameter(torch.randn(H, nq, nu) / nu ** 0.5)
        self.C2 = nn.Parameter(torch.randn(H, 1, nx) / nx ** 0.5)       # entre dans H (gain γ)
        self.D21 = nn.Parameter(0.1 * torch.randn(H, 1, nq) / nq ** 0.5)
        self.bx = nn.Parameter(torch.zeros(H, nx))
        self.bv = nn.Parameter(torch.zeros(H, nq))
        self.gate_proj = nn.Linear(nu, H)
        self.act = {"tanh": torch.tanh, "relu": torch.relu}[act]

    @property
    def n_state(self) -> int:
        return self.H * self.nx

    def frame(self) -> dict:
        H, nx, nq, N = self.H, self.nx, self.nq, self.N
        dt = self.X.dtype
        Hm = self.X.transpose(1, 2) @ self.X + self.eps * torch.eye(N, dtype=dt)
        K = torch.cat([self.C2, self.D21, torch.zeros(H, 1, nx, dtype=dt)], 2)             # (H,1,N)
        V = torch.cat([torch.zeros(H, nx, self.nu, dtype=dt), -self.D12, self.B2], 1)     # (H,N,nu)
        Hm = Hm + (K.transpose(1, 2) @ K + V @ V.transpose(1, 2)) / self.gamma
        H11, H21, H22 = Hm[:, :nx, :nx], Hm[:, nx:nx + nq, :nx], Hm[:, nx:nx + nq, nx:nx + nq]
        F, B1, P = Hm[:, nx + nq:, :nx], Hm[:, nx + nq:, nx:nx + nq], Hm[:, nx + nq:, nx + nq:]
        E = 0.5 * (H11 + P / self.alpha_bar ** 2 + self.Y - self.Y.transpose(1, 2))
        Einv = torch.linalg.inv(E)
        return dict(E=E, P=P, C1=-H21, lam=0.5 * H22.abs().sum(-1),
                    A=Einv @ F, Bw=Einv @ B1, Bu=Einv @ self.B2, bx=(Einv @ self.bx[..., None])[..., 0])

    def gate(self, u: Tensor) -> Tensor:
        return self.g_min + (1 - self.g_min) * torch.sigmoid(self.gate_proj(u))       # (B, H)

    def core(self, fr: dict, x: Tensor, u: Tensor) -> Tensor:
        """x : (B, H, nx), u : (B, nu) → f(x, u) : (B, H, nx)."""
        v = (torch.einsum("hqn,bhn->bhq", fr["C1"], x) + torch.einsum("hqm,bm->bhq", self.D12, u)
             + self.bv) / fr["lam"]
        w = self.act(v)
        return (torch.einsum("hij,bhj->bhi", fr["A"], x) + torch.einsum("hij,bhj->bhi", fr["Bw"], w)
                + torch.einsum("him,bm->bhi", fr["Bu"], u) + fr["bx"])

    def step(self, fr: dict, x: Tensor, u: Tensor, g: Tensor = None) -> Tensor:
        """Un pas. x : (B, H·nx) à plat ; retourne le nouvel état à plat."""
        B = x.shape[0]
        xs = x.view(B, self.H, self.nx)
        g = self.gate(u) if g is None else g
        xn = (1 - g[..., None]) * xs + g[..., None] * self.core(fr, xs, u)
        return xn.reshape(B, -1)
