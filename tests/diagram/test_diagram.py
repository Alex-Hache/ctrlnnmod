import torch
import pytest

from ctrlnmod.models.ssmodels.continuous.linear import SSLinear
from ctrlnmod.controllers import ILOFController, ILSFController
from ctrlnmod.blocks.primitives import Gain
from ctrlnmod.diagram import (
    Diagram, Series, FeedbackInterconnection, AlgebraicLoopError, WiringError,
)
from ctrlnmod.integrators import RK4Simulator


def _sim(model, u, x0, ts=1e-2):
    with torch.no_grad():
        return RK4Simulator(model, ts=ts)(u, x0)


def test_block_diagonal_is_exactly_decoupled():
    """Two independent blocks wired with no connections must reproduce running
    each block on its own input slice - exercises state slicing + IO mapping."""
    torch.manual_seed(0)
    a = SSLinear(1, 1, 2)
    b = SSLinear(1, 1, 3)
    diag = Diagram(
        blocks={"a": a, "b": b},
        connections=[],
        inputs=["a.u", "b.u"],
        outputs=["a.y", "b.y"],
    )
    assert diag.nu == 2 and diag.ny == 2 and diag.nx == 5
    u = torch.randn(4, 12, 2)
    x0 = torch.zeros(4, 5)
    _, y = _sim(diag, u, x0)
    _, ya = _sim(a, u[..., :1], torch.zeros(4, 2))
    _, yb = _sim(b, u[..., 1:], torch.zeros(4, 3))
    assert torch.allclose(y[..., :1], ya, atol=1e-6)
    assert torch.allclose(y[..., 1:], yb, atol=1e-6)


def test_series_runs_and_state_order():
    a = SSLinear(2, 2, 3)
    b = SSLinear(2, 2, 2)
    ser = Series([a, b])
    assert ser.nx == 5
    u = torch.randn(2, 6, 2)
    xs, ys = _sim(ser, u, torch.zeros(2, 5))
    assert xs.shape == (2, 6, 5)
    assert ys.shape == (2, 6, 2)


def test_algebraic_loop_detected_in_diagram():
    with pytest.raises(AlgebraicLoopError):
        Diagram(
            blocks={"g1": Gain(1), "g2": Gain(1)},
            connections=[("g1.out", "g2.in"), ("g2.out", "g1.in")],
            inputs=[],
            outputs=["g1.out"],
        )


def test_width_mismatch_detected():
    with pytest.raises(WiringError):
        Diagram(
            blocks={"a": SSLinear(1, 2, 2), "b": SSLinear(1, 1, 2)},
            connections=[("a.y", "b.u")],  # a.y width 2 != b.u width 1
            inputs=["a.u"],
            outputs=["b.y"],
        )


def test_gradients_flow_through_diagram():
    torch.manual_seed(0)
    a = SSLinear(1, 1, 2)
    b = SSLinear(1, 1, 2)
    ser = Series([a, b])
    u = torch.randn(2, 5, 1)
    _, y = RK4Simulator(ser, ts=1e-2)(u, torch.zeros(2, 4))
    y.pow(2).mean().backward()
    grads = [p.grad is not None for p in ser.parameters()]
    assert all(grads) and len(grads) > 0


def test_clone_is_independent():
    a = SSLinear(1, 1, 2)
    b = SSLinear(1, 1, 2)
    ser = Series([a, b])
    clone = ser.clone()
    assert isinstance(clone, Diagram)
    assert clone.nx == ser.nx and clone.nu == ser.nu and clone.ny == ser.ny
    # mutating the clone must not affect the original
    with torch.no_grad():
        for p in clone.parameters():
            p.add_(1.0)
    diffs = [not torch.allclose(p, q) for p, q in zip(ser.parameters(), clone.parameters())]
    assert any(diffs)


def test_feedback_output_and_state_feedback_build_and_run():
    # output feedback
    plant = SSLinear(1, 1, 2)
    ctrl = ILOFController(nu=1, ny=1, hidden_layers=[8])
    fb = FeedbackInterconnection(plant, ctrl)
    u = torch.randn(2, 5, 1)
    xs, ys = _sim(fb, u, torch.zeros(2, fb.nx), ts=1e-3)
    assert ys.shape == (2, 5, 1)

    # state feedback uses a state tap
    plant2 = SSLinear(1, 1, 2)
    ctrl2 = ILSFController(nu=1, nx=2, hidden_layers=[8])
    fb2 = FeedbackInterconnection(plant2, ctrl2)
    assert len(fb2.state_taps) == 1
    xs2, ys2 = _sim(fb2, u, torch.zeros(2, fb2.nx), ts=1e-3)
    assert ys2.shape == (2, 5, 1)
