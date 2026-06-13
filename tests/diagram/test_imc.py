import torch
import pytest

from ctrlnmod.models.ssmodels.continuous.linear import SSLinear
from ctrlnmod.controllers import NeuralController
from ctrlnmod.diagram import IMC, Diagram
from ctrlnmod.integrators import RK4Simulator
from ctrlnmod.losses import ReferenceTrackingLoss


def _stable_ss(seed):
    """A stable, controllable SISO linear model with hand-set weights (no cvxpy)."""
    torch.manual_seed(seed)
    m = SSLinear(1, 1, 2)
    with torch.no_grad():
        m.A.weight.copy_(torch.tensor([[-1.0, 0.5], [0.0, -2.0]]))
        m.B.weight.copy_(torch.tensor([[1.0], [1.0]]))
        m.C.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return m


def _build_imc(seed=0):
    torch.manual_seed(seed)
    plant = _stable_ss(seed)
    model = _stable_ss(seed + 100)
    return IMC(plant, model, hidden_layers=[16])


def test_imc_builds_and_simulates():
    imc = _build_imc()
    assert isinstance(imc, Diagram)
    assert imc.nu == 1 and imc.ny == 1
    assert imc.nx == imc.plant.nx + imc.model.nx
    r = torch.randn(3, 10, 1)
    xs, ys = RK4Simulator(imc, ts=1e-2)(r, torch.zeros(3, imc.nx))
    assert xs.shape == (3, 10, imc.nx)
    assert ys.shape == (3, 10, 1)
    assert ys.requires_grad


def test_imc_no_algebraic_loop_and_trainable_selection():
    imc = _build_imc()
    imc.trainable_blocks("controller", "model")
    assert not any(p.requires_grad for p in imc.plant.parameters())
    assert all(p.requires_grad for p in imc.controller.parameters())
    assert all(p.requires_grad for p in imc.model.parameters())


def test_imc_clone_is_imc():
    imc = _build_imc()
    c = imc.clone()
    assert isinstance(c, IMC)
    assert c.nx == imc.nx


def test_imc_training_reduces_tracking_loss():
    """A few optimisation steps on the controller should reduce a setpoint-tracking
    loss - validates that the closed loop is differentiable end-to-end."""
    imc = _build_imc(seed=1)
    imc.trainable_blocks("controller")  # learn Q only, plant & model fixed
    sim = RK4Simulator(imc, ts=2e-2)

    # step reference; goal: closed-loop output should follow r
    batch, T = 4, 40
    r = torch.ones(batch, T, 1)
    x0 = torch.zeros(batch, imc.nx)
    criterion = ReferenceTrackingLoss()

    opt = torch.optim.Adam([p for p in imc.parameters() if p.requires_grad], lr=5e-3)
    _, y0 = sim(r, x0)
    loss0 = criterion(y0, r).item()
    for _ in range(40):
        opt.zero_grad()
        _, y = sim(r, x0)
        loss = criterion(y, r)
        loss.backward()
        opt.step()
    _, yf = sim(r, x0)
    lossf = criterion(yf, r).item()
    assert lossf < loss0, f"tracking loss did not decrease: {loss0:.4f} -> {lossf:.4f}"
