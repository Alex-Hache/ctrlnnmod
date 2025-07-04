import pytest
import torch
from torch import Tensor
from ctrlnmod.models.ssmodels.continuous.linear import SSLinear, ExoSSLinear
from ctrlnmod.lmis.hinf import HInfCont


@pytest.fixture
def dims():
    return {
        'nu': 1,
        'ny': 2,
        'nx': 4,
        'nd': 1
    }

@pytest.fixture
def build_linear(dims):
    return SSLinear(dims['nu'], dims['ny'], dims['nx'], alpha=1e-2)

@pytest.fixture
def build_exo_linear(dims):
    return ExoSSLinear(dims['nu'], dims['ny'], dims['nx'], dims['nd'], alpha=1e-2)


@pytest.mark.parametrize("builder", ["build_linear", "build_exo_linear"])
def test_hinf_lmi_hook(request, builder, dims):
    model = request.getfixturevalue(builder)

    lmi = HInfCont(model, extract_matrices=model.to_hinf)

    # Trigger the forward_pre_hook via dummy forward
    M_forward, P_forward = lmi()

    # Compare the M sovled and the M obtained via forward
    M_solve, gamma_solve, P_solve = HInfCont.solve(lmi.A.detach(), lmi.B.detach(), 
                                                      lmi.C.detach(), torch.zeros(dims['ny'], dims['nu']))

    assert torch.dist(lmi.A, model.A.weight) < 0.1
    if 'Exo' in model.__class__.__name__:
        assert torch.dist(lmi.B, model.G.weight) < 0.1
    else:
        assert torch.dist(lmi.B, model.B.weight) < 0.1

    assert torch.dist(lmi.C, model.C.weight) < 0.1
    assert torch.dist(P_forward, P_solve) < 0.1
    assert lmi.P is not None, "P should be initialized by the hook"
    assert lmi.gamma is not None, "L2 gain should be initialized by the hook"
