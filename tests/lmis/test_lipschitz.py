import pytest
import torch
from torch import Tensor
from ctrlnmod.models.feedforward import FFNN, LBDN
from ctrlnmod.lmis.lipschitz import Lipschitz


@pytest.fixture
def dims():
    return {
        'n_in': 4,
        'n_out': 2,
        'hidden_layers': [6]
    }

@pytest.fixture
def build_ffnn(dims):
    return FFNN(dims['n_in'], dims['hidden_layers'], dims['n_out'])

@pytest.fixture
def build_lbdn(dims):
    gamma = torch.Tensor([2.0])
    return LBDN(dims['n_in'], dims['hidden_layers'], dims['n_out'], scale=gamma)


@pytest.mark.parametrize("builder", ["build_ffnn", "build_lbdn"])
def test_lipschitz_lmi_hook(request, builder):
    model = request.getfixturevalue(builder)

    lmi = Lipschitz(model, extract_matrices=model.to_snof)

    # Trigger the forward_pre_hook via dummy forward
    M_forward, Lambda_forward = lmi()

    # Compare the M sovled and the M obtained via forward
    M_solve, lip_solve, Lambda_solve = Lipschitz.solve(lmi.A.detach(), lmi.B.detach(), lmi.C.detach())

    assert lmi.lip == pytest.approx(lip_solve, rel=1e-4)
    assert torch.dist(M_solve, M_forward) < 1e-4
    assert torch.dist(Lambda_solve, Lambda_forward) < 1e-4
    assert lmi.Lambda_vec is not None, "Lambda_vec should be initialized by the hook"
    assert lmi.lip is not None, "Lipschitz constant should be initialized by the hook"
