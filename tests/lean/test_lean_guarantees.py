"""
Tests numériques des garanties prouvées en Lean (dépôt `these-lean`).

Chaque test vérifie, sur des paramètres tirés au hasard, la propriété que le théorème
Lean correspondant garantit. Un échec signifie que le code s'écarte de l'énoncé prouvé.

Correspondance (noms Lean dans l'espace de noms `These`) :
- AlphaStable                 → thm_3_3_5_corrected, thm_3_3_6
- L2BoundedLinear('sqrtm')    → ms_variant_sound, ms_variant_complete
- L2BoundedLinear('riccati')  → thm_3_3_8_riccati, thm_3_3_8_image_not_hurwitz
- H2Linear                    → thm_3_3_12_gramian, thm_3_3_12_trace
- ContractingRENODE           → thm_1_6_10 (LMI (1.95) / (3.100)), explicit_wellPosed
- SandwichLayer, LBDN         → sandwich_nonexpansive, thm_3_4_2, prop_3_4_3
"""
import numpy as np
import pytest
import scipy.linalg as sl
import torch

torch.set_default_dtype(torch.float64)

N_DRAWS = 20


def hinf_norm(A, B, C, n_freq=2000):
    """Norme H∞ estimée par balayage fréquentiel (borne inférieure de la vraie norme)."""
    A, B, C = (np.asarray(M, dtype=float) for M in (A, B, C))
    n = A.shape[0]
    w = np.concatenate([[0.0], np.logspace(-4, 4, n_freq)])
    return max(np.linalg.norm(C @ np.linalg.solve(1j * x * np.eye(n) - A, B), 2) for x in w)


def max_real_eig(A):
    return float(np.max(np.linalg.eigvals(np.asarray(A, dtype=float)).real))


# ----------------------------------------------------------------------------------------
# α-stabilité
# ----------------------------------------------------------------------------------------

def test_alpha_stable_sound():
    """thm_3_3_5_corrected (⟸) : A = P(−Q/2 + S) − αI vérifie ℜλ < −α."""
    from ctrlnmod.models.ssmodels.continuous.linear.base import SSLinear
    for seed in range(N_DRAWS):
        torch.manual_seed(seed)
        alpha = 0.5
        model = SSLinear(2, 2, 4, alpha=alpha)
        A, _, _ = model._frame()
        assert max_real_eig(A.detach()) < -alpha


def test_sslinear_check_sign():
    """`SSLinear.check_` doit refuser une matrice qui n'est pas α-stable.

    Avec α = 1, la matrice A = −0.5 I a ℜλ = −0.5 > −1 : elle n'est pas 1-stable.
    """
    from ctrlnmod.models.ssmodels.continuous.linear.base import SSLinear
    model = SSLinear(1, 1, 2, alpha=None)
    model.alpha = 1.0
    model.A.weight = torch.nn.Parameter(-0.5 * torch.eye(2))
    ok, _ = model.check_()
    assert not ok, "check_ accepte une matrice qui n'est pas 1-stable (signe de alpha)"


# ----------------------------------------------------------------------------------------
# H∞ : variante M_s (param='sqrtm')
# ----------------------------------------------------------------------------------------

def test_hinf_sqrtm_sound():
    """ms_variant_sound : tout point de la paramétrisation a un gain < γ et A Hurwitz."""
    from ctrlnmod.models.ssmodels.continuous.linear.hinf import L2BoundedLinear
    gamma = 0.7
    for seed in range(N_DRAWS):
        torch.manual_seed(seed)
        model = L2BoundedLinear(2, 3, 4, gamma=gamma, param='sqrtm')
        A, B, C = (M.detach() for M in model._frame())
        assert max_real_eig(A) < 0
        assert hinf_norm(A, B, C) < gamma


@pytest.mark.xfail(strict=True, reason="init : terme eps*I absent de S et gamma remplacé par le gain "
                   "du système ; construction exacte dans ms_variant_complete.")
def test_hinf_sqrtm_right_inverse_exact():
    """ms_variant_complete : l'initialisation doit redonner exactement (A, B, C).

    Système stable avec gain ≈ 0.77 < γ = 1.5 et rang B = n_u.
    """
    from ctrlnmod.models.ssmodels.continuous.linear.hinf import L2BoundedLinear
    rng = np.random.default_rng(0)
    nx, nu, ny = 4, 2, 3
    A0 = rng.normal(size=(nx, nx))
    A0 = A0 - (max_real_eig(A0) + 1.0) * np.eye(nx)
    B0 = rng.normal(size=(nx, nu)) * 0.3
    C0 = rng.normal(size=(ny, nx)) * 0.3
    gamma = 1.5
    assert hinf_norm(A0, B0, C0) < gamma
    model = L2BoundedLinear(nu, ny, nx, gamma=gamma, param='sqrtm')
    model.init_weights_(torch.tensor(A0), torch.tensor(B0), torch.tensor(C0), gamma, 0.0)
    A, B, C = (M.detach().numpy() for M in model._frame())
    err = max(np.abs(A - A0).max(), np.abs(B - B0).max(), np.abs(C - C0).max())
    assert err < 1e-6, f"reconstruction inexacte : erreur max = {err:.2e}"


@pytest.mark.xfail(strict=True, reason="init : self.gamma = gamma_sys (bord de l'ensemble). À décider.")
def test_hinf_sqrtm_keeps_prescribed_gamma():
    """Après initialisation, la borne garantie doit rester le γ demandé, pas le gain du
    système initial (sinon l'entraînement est restreint au bord de l'ensemble)."""
    from ctrlnmod.models.ssmodels.continuous.linear.hinf import L2BoundedLinear
    rng = np.random.default_rng(1)
    nx, nu, ny = 3, 1, 1
    A0 = -np.eye(nx) + 0.1 * rng.normal(size=(nx, nx))
    B0 = 0.2 * rng.normal(size=(nx, nu))
    C0 = 0.2 * rng.normal(size=(ny, nx))
    gamma = 2.0
    model = L2BoundedLinear(nu, ny, nx, gamma=gamma, param='sqrtm')
    model.init_weights_(torch.tensor(A0), torch.tensor(B0), torch.tensor(C0), gamma, 0.0)
    assert float(model.gamma) == pytest.approx(gamma), (
        f"γ remplacé par le gain initial {float(model.gamma):.3f} au lieu de {gamma}")


# ----------------------------------------------------------------------------------------
# H∞ : paramétrisation par Riccati (Th. 3.3.8)
# ----------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Th. 3.3.8 : Riccati avec P > 0 ne garantit pas A Hurwitz "
                   "(thm_3_3_8_image_not_hurwitz). Ajouter une marge (alpha > 0) ou une garde.")
def test_hinf_riccati_can_be_unstable():
    """thm_3_3_8_image_not_hurwitz : avec G = 0 et B = 0, A = S P n'est pas Hurwitz.

    La paramétrisation Riccati ne garantit pas la stabilité à elle seule.
    Ce test documente le comportement ; il échoue tant qu'aucune garde n'existe.
    """
    from ctrlnmod.models.ssmodels.continuous.linear.hinf import L2BoundedLinear
    model = L2BoundedLinear(1, 1, 2, gamma=1.0, param='riccati')
    with torch.no_grad():
        model.B.zero_()
        model.G.zero_()
    A, _, _ = (M.detach() for M in model._frame())
    assert max_real_eig(A) < -1e-9, "Riccati satisfaite mais A non Hurwitz (Lemme 3.3.7)"


# ----------------------------------------------------------------------------------------
# H2 (Th. 3.3.12, 3.3.13)
# ----------------------------------------------------------------------------------------

def test_h2_sound():
    """thm_3_3_12_gramian + thm_3_3_12_trace : norme H2 exactement égale à γ₂."""
    from ctrlnmod.models.ssmodels.continuous.linear.h2 import H2Linear
    gamma2 = 0.8
    for seed in range(N_DRAWS):
        torch.manual_seed(seed)
        model = H2Linear(2, 2, 4, gamma2)
        A, B, C = (M.detach().numpy() for M in model._frame())
        if max_real_eig(A) >= 0:
            continue  # (C, A) non observable : voir thm_3_3_12_needs_minimality
        Wc = sl.solve_continuous_lyapunov(A, -B @ B.T)
        h2 = np.sqrt(np.trace(C @ Wc @ C.T))
        assert h2 == pytest.approx(gamma2, rel=1e-6)


# ----------------------------------------------------------------------------------------
# REN continu contractant (Th. 1.6.10, Th. 3.5.1, Th. 3.5.2)
# ----------------------------------------------------------------------------------------

def _lure_lmi(A, B1, C1, D11, P, Lam, alpha):
    """LMI (1.95) avec marge α : doit être définie négative."""
    M11 = A.T @ P + P @ A + 2 * alpha * P
    M12 = P @ B1 + C1.T @ Lam
    M22 = -2 * Lam + Lam @ D11 + D11.T @ Lam
    return np.block([[M11, M12], [M12.T, M22]])


def test_contracting_renode_sound():
    """thm_1_6_10 : la LMI de Lur'e est vérifiée avec Γ = Λ, et D₁₁ est strictement
    triangulaire (réseau explicite, explicit_wellPosed)."""
    from ctrlnmod.models.ssmodels.continuous.general.renode import ContractingRENODE
    for seed in range(N_DRAWS):
        torch.manual_seed(seed)
        model = ContractingRENODE(2, 2, 3, 4, alpha=0.1, param='square')
        with torch.no_grad():
            for p in (model.X, model.P_inv, model.S, model.U):
                p.copy_(torch.randn_like(p))
        A, B1, _, C1, _, D11, *_ = (M.detach().numpy() if M is not None else None
                                    for M in model._frame())
        P_inv = (model.P_inv @ model.P_inv.T).detach().numpy()
        P = np.linalg.inv(P_inv)
        Lam = model.Lambda.numpy()
        M = _lure_lmi(A, B1, C1, D11, P, Lam, float(model.alpha))
        assert np.max(np.linalg.eigvalsh((M + M.T) / 2)) < 0
        assert np.allclose(np.triu(D11), 0)


def test_contracting_renode_check_uses_lambda():
    """`check()` doit tester la même LMI que le théorème (bloc PB₁ + C₁ᵀΛ)."""
    from ctrlnmod.models.ssmodels.continuous.general.renode import ContractingRENODE
    torch.manual_seed(0)
    model = ContractingRENODE(2, 2, 3, 4, alpha=0.1, param='square')
    with torch.no_grad():
        for p in (model.X, model.P_inv, model.S, model.U):
            p.copy_(torch.randn_like(p))
    ok, _ = model.check()
    assert ok, "check() rejette un modèle qui satisfait la LMI (Λ remplacé par diag(D₁₁))"


# ----------------------------------------------------------------------------------------
# Réseaux Lipschitz (Th. 3.4.2, Prop. 3.4.3)
# ----------------------------------------------------------------------------------------

def test_lbdn_lipschitz():
    """thm_3_4_2 + prop_3_4_3 : ‖f(x) − f(x')‖ ≤ ‖diag(scale)(x − x')‖."""
    from ctrlnmod.models.feedforward.lbdn import LBDN
    for seed in range(N_DRAWS):
        torch.manual_seed(seed)
        scale = torch.tensor([0.5, 2.0, 1.0])
        net = LBDN(3, [16, 16], 2, scale=scale, act_f='tanh')
        with torch.no_grad():
            for layer in net.layers:
                if hasattr(layer, 'psi'):
                    layer.psi.copy_(torch.randn_like(layer.psi))
            x = torch.randn(200, 3)
            xp = x + 0.3 * torch.randn(200, 3)
            lhs = (net(x) - net(xp)).norm(dim=1)
            rhs = ((x - xp) * scale).norm(dim=1)
        assert torch.all(lhs <= rhs + 1e-9)
