import pytest
import torch
import torch.nn as nn
from ctrlnmod.regularizations import L1Regularization, L2Regularization


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

    def forward(self, x):
        return self.linear(x)


class TestLRegularizations:

    # --- L1 Tests ---
    def test_build_l1_regularization(self):
        model = DummyModel()
        reg = L1Regularization(model, lambda_l1=0.1, update_factor=0.01)
        assert isinstance(reg, L1Regularization)
        assert reg.lambda_l1.item() == pytest.approx(0.1, rel=1e-6)

    def test_call_l1_regularization(self):
        model = DummyModel()
        reg = L1Regularization(model, lambda_l1=0.1, update_factor=0.01)
        expected = 0.1 * (1 + 2 + 3 + 4)  # L1 norm = 10
        penalty = reg()
        assert penalty.item() == pytest.approx(expected, rel=1e-6)

    def test_update_l1_regularization(self):
        model = DummyModel()
        reg = L1Regularization(model, lambda_l1=0.1, update_factor=0.1, updatable=True)
        original = model.linear.weight.data.clone()
        reg.update()
        expected = original - 0.1 * original.sign()
        assert torch.allclose(model.linear.weight.data, expected, atol=1e-6)

    def test_update_l1_not_updatable(self):
        model = DummyModel()
        reg = L1Regularization(model, lambda_l1=0.1, update_factor=0.1, updatable=False)
        original = model.linear.weight.data.clone()
        reg.update()
        assert torch.allclose(model.linear.weight.data, original, atol=1e-6)

    # --- L2 Tests ---
    def test_build_l2_regularization(self):
        model = DummyModel()
        reg = L2Regularization(model, lambda_l2=0.01, update_factor=0.05)
        assert isinstance(reg, L2Regularization)
        assert reg.lambda_l2.item() == pytest.approx(0.01, rel=1e-6)

    def test_call_l2_regularization(self):
        model = DummyModel()
        reg = L2Regularization(model, lambda_l2=0.01, update_factor=0.05)
        # L2 norm = 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
        expected = 0.01 * 30.0
        penalty = reg()
        assert penalty.item() == pytest.approx(expected, rel=1e-6)

    def test_update_l2_regularization(self):
        model = DummyModel()
        reg = L2Regularization(model, lambda_l2=0.01, update_factor=0.1, updatable=True)
        original = model.linear.weight.data.clone()
        reg.update()
        expected = original - 0.1 * original
        assert torch.allclose(model.linear.weight.data, expected, atol=1e-6)

    def test_update_l2_not_updatable(self):
        model = DummyModel()
        reg = L2Regularization(model, lambda_l2=0.01, update_factor=0.1, updatable=False)
        original = model.linear.weight.data.clone()
        reg.update()
        assert torch.allclose(model.linear.weight.data, original, atol=1e-6)
