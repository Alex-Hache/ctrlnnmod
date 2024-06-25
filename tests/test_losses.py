import unittest
import torch
import torch.nn as nn
from ..ctrl_nmod.losses import MSELoss, NMSELoss, FitPercentLoss, RMSELoss, NRMSELoss
from ..ctrl_nmod.regularizations import L1Regularization


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestLosses(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.regularizers = [L1Regularization(
            self.model, lambda_l1=0.01, update_factor=0.1)]
        self.criterion_mse = MSELoss(self.regularizers)
        self.criterion_nmse = NMSELoss(self.regularizers)
        self.criterion_fitpercent = FitPercentLoss(self.regularizers)
        self.criterion_rmse = RMSELoss(self.regularizers)
        self.criterion_nrmse = NRMSELoss(self.regularizers)
        self.output = torch.randn(10, 1)
        self.target = torch.randn(10, 1)

    def test_mse_loss_with_regularization(self):
        base_loss = torch.mean((self.output - self.target) ** 2)
        loss_value = self.criterion_mse(self.output, self.target)
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.item() >= base_loss.item())

    def test_nmse_loss_with_regularization(self):
        base_loss = torch.mean((self.output - self.target) ** 2) / torch.mean(self.target ** 2)
        loss_value = self.criterion_nmse(self.output, self.target)
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.item() >= base_loss.item())

    def test_fitpercent_loss_with_regularization(self):
        loss_value = self.criterion_fitpercent(self.output, self.target)
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(0 <= loss_value.item() <= 1)

    def test_rmse_loss_with_regularization(self):
        base_loss = torch.sqrt(torch.mean((self.output - self.target) ** 2))
        loss_value = self.criterion_rmse(self.output, self.target)
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.item() >= base_loss.item())

    def test_nrmse_loss_with_regularization(self):
        base_loss = torch.sqrt(torch.mean(
            (self.output - self.target) ** 2)) / torch.std(self.target)
        loss_value = self.criterion_nrmse(self.output, self.target)
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertTrue(loss_value.item() >= base_loss.item())


if __name__ == '__main__':
    unittest.main()
