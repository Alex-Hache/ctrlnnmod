import torch
import pytest

from ctrlnmod.controllers import (
    Controller,
    NeuralController,
    ILOFController,
    ILSFController,
    BetaILSFController,
)


@pytest.mark.parametrize("nd", [None, 1])
def test_ilof_is_controller_and_evaluate_matches_forward(nd):
    torch.manual_seed(0)
    c = ILOFController(nu=2, ny=3, hidden_layers=[8], nd=nd)
    assert isinstance(c, Controller)
    assert c.out_ports == {"v": 2}
    assert ("d" in c.in_ports) == (nd is not None)
    u = torch.randn(4, 2)
    y = torch.randn(4, 3)
    d = torch.randn(4, 1) if nd else None
    inp = {"u": u, "y": y}
    if d is not None:
        inp["d"] = d
    v_eval = c.evaluate(inp)["v"]
    v_fwd = c(u, y, d)
    assert torch.allclose(v_eval, v_fwd)


def test_ilsf_state_feedback_ports():
    c = ILSFController(nu=2, nx=4, hidden_layers=[8])
    assert "x" in c.in_ports and c.in_ports["x"] == 4
    out = c.evaluate({"u": torch.zeros(3, 2), "x": torch.randn(3, 4)})
    assert out["v"].shape == (3, 2)


def test_clone_is_independent():
    c = ILSFController(nu=1, nx=2, hidden_layers=[8])
    c2 = c.clone()
    assert isinstance(c2, ILSFController)
    # parameters copied
    for p, p2 in zip(c.parameters(), c2.parameters()):
        assert torch.allclose(p, p2)


def test_neural_controller_maps_error_to_control():
    c = NeuralController(n_in=3, n_out=2, hidden_layers=[16], act_f="tanh")
    assert c.in_ports == {"e": 3} and c.out_ports == {"u": 2}
    out = c.evaluate({"e": torch.randn(5, 3)})
    assert out["u"].shape == (5, 2)
    c2 = c.clone()
    assert isinstance(c2, NeuralController)


def test_beta_state_feedback_builds():
    c = BetaILSFController(nu=2, nx=3, hidden_layers=[8])
    out = c.evaluate({"u": torch.zeros(4, 2), "x": torch.randn(4, 3)})
    assert out["v"].shape == (4, 2)
