import pytest
import torch
from torch import Tensor
from ctrlnmod.regularizations import LogdetRegularization
from ctrlnmod.models.feedforward import FFNN, LBDN
from ctrlnmod.lmis.lipschitz import Lipschitz


@pytest.fixture
def dims():
    return {
        'n_in': 10,
        'n_out': 4,
        'hidden_layers': [16, 8, 4]
    }

@pytest.mark.parametrize("model_class", [FFNN, LBDN])
def test_logdet_regularization_models(dims, model_class):
    gamma = torch.Tensor([2.0])
    if model_class is LBDN:
        model = model_class(dims['n_in'], dims['hidden_layers'], dims['n_out'], scale=gamma)
    else:
        model = model_class(dims['n_in'], dims['hidden_layers'], dims['n_out'])

    # Build Lipschitz LMI
    lmi = Lipschitz(model, model.to_snof)

    # Regularization setup
    reg = LogdetRegularization(lmi, lambda_logdet=0.5, update_factor=0.9)

    # === Assertions ===
    assert isinstance(reg.lambda_logdet, Tensor)
    assert reg.lambda_logdet.item() == pytest.approx(0.5, rel=1e-6)
    assert reg.min_weight == pytest.approx(1e-6)

    # === Run regularization computation ===
    reg_value = reg()

    assert torch.isfinite(reg_value), "Regularization returned NaN or Inf"
    reg_value.backward()  # Check gradient flow
