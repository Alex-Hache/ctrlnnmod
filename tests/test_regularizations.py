import unittest
import torch
import torch.nn as nn
from ..ctrl_nmod.regularizations import L1Regularization, L2Regularization, LogdetRegularization, StateRegularization


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestRegularizations(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.x = torch.randn(10)
        self.y = torch.randn(10)

    def test_l1_regularization(self):
        l1_reg = L1Regularization(
            self.model, lambda_l1=0.01, update_factor=0.1)
        reg_value = l1_reg()
        self.assertIsInstance(reg_value, torch.Tensor)
        self.assertTrue(reg_value.item() >= 0)

    def test_l2_regularization(self):
        l2_reg = L2Regularization(
            self.model, lambda_l2=0.01, update_factor=0.1)
        reg_value = l2_reg()
        self.assertIsInstance(reg_value, torch.Tensor)
        self.assertTrue(reg_value.item() >= 0)

    '''
    def test_logdet_regularization(self):
        lmi_model = nn.Linear(10, 10)
        logdet_reg = LogdetRegularization(
            lmi_model, lambda_logdet=0.01, update_factor=0.1)
        with self.assertRaises(ValueError):
            logdet_reg()
    '''

    def test_state_regularization(self):
        state_reg = StateRegularization(
            self.model, lambda_state=0.01, update_factor=0.1)
        reg_value = state_reg(self.x, self.y)
        self.assertIsInstance(reg_value, torch.Tensor)
        self.assertTrue(reg_value.item() >= 0)


if __name__ == '__main__':
    unittest.main()
