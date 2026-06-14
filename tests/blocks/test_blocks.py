import torch
import pytest

from ctrlnmod.blocks import Block, Sum, Gain, Constant, Saturation
from ctrlnmod.models.ssmodels.continuous.linear import SSLinear


def test_sum_signed():
    s = Sum(2, signs=["+", "-"])
    assert s.in_ports == {"in0": 2, "in1": 2}
    assert s.out_ports == {"out": 2}
    out = s.evaluate({"in0": torch.ones(3, 2), "in1": torch.full((3, 2), 0.25)})
    assert torch.allclose(out["out"], torch.full((3, 2), 0.75))


def test_gain_matrix_and_scalar():
    g = Gain(2, gain=3.0)
    out = g.evaluate({"in": torch.ones(4, 2)})
    assert torch.allclose(out["out"], torch.full((4, 2), 3.0))
    K = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    gm = Gain(2, 2, gain=K)
    out = gm.evaluate({"in": torch.tensor([[1.0, 1.0]])})
    assert torch.allclose(out["out"], torch.tensor([[3.0, 1.0]]))


def test_constant_broadcasts_batch():
    c = Constant([1.0, 2.0])
    out = c.evaluate({}, batch_size=5)
    assert out["out"].shape == (5, 2)
    assert torch.allclose(out["out"][0], torch.tensor([1.0, 2.0]))


def test_saturation():
    sat = Saturation(1, low=-0.5, high=0.5)
    out = sat.evaluate({"in": torch.tensor([[-1.0], [0.0], [1.0]])})
    assert torch.allclose(out["out"], torch.tensor([[-0.5], [0.0], [0.5]]))


def test_clone_independent():
    g = Gain(2, gain=2.0, trainable=True)
    g2 = g.clone()
    with torch.no_grad():
        g2.K.add_(1.0)
    assert not torch.allclose(g.K, g2.K)


def test_ssmodel_is_block_with_ports():
    m = SSLinear(2, 3, 4)
    assert isinstance(m, Block)
    assert m.is_dynamic and m.nx == 4
    assert m.in_ports == {"u": 2} and m.out_ports == {"y": 3}
    assert m.feedthrough is False


def test_ssmodel_dynamics_and_eval_output_agree():
    torch.manual_seed(0)
    m = SSLinear(2, 3, 4)
    x = torch.randn(5, 4)
    u = torch.randn(5, 2)
    dx, outs = m.dynamics({"u": u}, x)
    dx_ref, y_ref = m(u, x)
    assert torch.allclose(outs["y"], y_ref)
    assert torch.allclose(dx, dx_ref)
    # strictly proper: output independent of u
    eo = m.eval_output(x)
    assert torch.allclose(eo["y"], y_ref)
