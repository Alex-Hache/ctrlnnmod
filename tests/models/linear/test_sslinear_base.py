import torch
import pytest
from ctrlnmod.models.ssmodels.continuous import SSLinear, ExoSSLinear

@pytest.fixture
def dims():
    return {"nu": 2, "ny": 3, "nx": 4, "nd": 2}

@pytest.fixture
def tensors(dims):
    A = torch.eye(dims["nx"])
    B = torch.ones(dims["nx"], dims["nu"])
    C = torch.ones(dims["ny"], dims["nx"])
    G = torch.ones(dims["nx"], dims["nd"])
    return A, B, C, G

def test_sslinear_forward_shapes(dims):
    model = SSLinear(dims["nu"], dims["ny"], dims["nx"])
    u = torch.randn(5, dims["nu"])
    x = torch.randn(5, dims["nx"])
    dx, y = model(u, x)
    assert dx.shape == (5, dims["nx"])
    assert y.shape == (5, dims["ny"])

def test_exosslinear_forward_shapes(dims):
    model = ExoSSLinear(dims["nu"], dims["ny"], dims["nx"], dims["nd"])
    u = torch.randn(5, dims["nu"])
    x = torch.randn(5, dims["nx"])
    d = torch.randn(5, dims["nd"])
    dx, y = model(u, x, d)
    assert dx.shape == (5, dims["nx"])
    assert y.shape == (5, dims["ny"])

def test_right_inverse_and_eval(dims, tensors):
    A, B, C, _ = tensors
    model = SSLinear(dims["nu"], dims["ny"], dims["nx"])
    model._right_inverse(A, B, C)
    A_eval, B_eval, C_eval = model._frame()
    assert torch.allclose(A_eval, A)
    assert torch.allclose(B_eval, B)
    assert torch.allclose(C_eval, C)

def test_check_stability(dims):
    A = -0.5 * torch.eye(dims["nx"])
    model = SSLinear(dims["nu"], dims["ny"], dims["nx"], alpha=0.0)
    model._right_inverse(A, torch.zeros(dims["nx"], dims["nu"]), torch.zeros(dims["ny"], dims["nx"]))
    stable, max_eig = model.check_()
    assert stable is True
    assert max_eig <= 0.0

def test_clone_behavior_sslinear(dims, tensors):
    A, B, C, _ = tensors
    model = SSLinear(dims["nu"], dims["ny"], dims["nx"])
    model._right_inverse(A, B, C)
    model_clone = model.clone()
    for p1, p2 in zip(model.parameters(), model_clone.parameters()):
        assert torch.allclose(p1, p2)

def test_clone_behavior_exosslinear(dims, tensors):
    A, B, C, G = tensors
    model = ExoSSLinear(dims["nu"], dims["ny"], dims["nx"], dims["nd"])
    model._right_inverse(A, B, C, G)
    model_clone = model.clone()
    for p1, p2 in zip(model.parameters(), model_clone.parameters()):
        assert torch.allclose(p1, p2)

def test_frame_and_cache(dims, tensors):
    A, B, C, G = tensors
    model = ExoSSLinear(dims["nu"], dims["ny"], dims["nx"], dims["nd"])
    model._frame_cache.is_caching = True
    model._right_inverse(A, B, C, G)
    frame1 = model._frame()
    frame2 = model._frame()
    assert all(torch.equal(f1, f2) for f1, f2 in zip(frame1, frame2))

def test_init_weights_alpha_adjust(dims, tensors):
    _, B, C, G = tensors
    A = -0.3 * torch.eye(dims["nx"])
    model = ExoSSLinear(dims["nu"], dims["ny"], dims["nx"], dims["nd"], alpha=0.0)
    model.init_weights_(A, B, C, G)
    eval_weights = model.eval_()

    print("Eval Weights:",eval_weights[0])
    print("Expected Weights:", A)
    assert torch.allclose(eval_weights[0], A, atol=1e-4)

def test_repr_and_str(dims):
    model = SSLinear(dims["nu"], dims["ny"], dims["nx"], alpha=0.5)
    assert isinstance(repr(model), str)
    assert str(model) == repr(model)
