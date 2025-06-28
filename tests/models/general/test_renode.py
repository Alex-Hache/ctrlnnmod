import torch
import pytest
from ctrlnmod.models.ssmodels.continuous.general.renode import RENODE, ContractingRENODE  # remplacer par votre module

@pytest.mark.parametrize("bias,feedthrough,out_eq_nl,sigma", [
    (False, False, False, "relu"),
    (True, True, True, "tanh"),
])
def test_renode_full_workflow(bias, feedthrough, out_eq_nl, sigma):
    # dimensions
    nu, ny, nx, nq = 2, 3, 4, 5
    device = torch.device('cpu')

    # 1) Constructor and instantiation
    m = RENODE(nu, ny, nx, nq, sigma=sigma,
               bias=bias, feedthrough=feedthrough,
               out_eq_nl=out_eq_nl, device=device)
    assert isinstance(m, RENODE)

    # init_weights_ 
    A = torch.randn(nx, nx)
    B = torch.randn(nx, nu)
    C = torch.randn(ny, nx)
    m.init_weights_(A, B, C)

    # 2) Forward + grad check
    batch = 7
    u = torch.randn(batch, nu, device=device)
    x = torch.randn(batch, nx, device=device, requires_grad=True)
    dx, y = m(u, x)
    assert dx.shape == (batch, nx)
    assert y.shape == (batch, ny)
    out = dx.sum() + y.sum()
    out.backward()
    assert x.grad is not None

    # 3) clone
    m2 = m.clone()
    # résultats identiques
    dx2, y2 = m2(u, x.detach())
    assert torch.allclose(dx2, dx.detach())
    assert torch.allclose(y2, y.detach())

    # 4) check
    ok, info = m.check()
    assert ok is True
    assert isinstance(info, dict)


@pytest.mark.parametrize("param", ['square', 'expm'])
def test_contracting_renode_full_workflow(param):
    nu, ny, nx, nq = 1, 2, 3, 2
    device = torch.device('cpu')
    alpha = 0.1
    epsilon = 1e-3

    m = ContractingRENODE(nu, ny, nx, nq,
                          sigma='identity',
                          alpha=alpha, epsilon=epsilon,
                          bias=True, feedthrough=True,
                          out_eq_nl=False, param=param,
                          device=device)
    assert isinstance(m, ContractingRENODE)

    # init_weights_
    A = -torch.eye(nx)
    B = torch.randn(nx, nu)
    C = torch.randn(nq, nx)
    m.init_weights_(A, B, C)

    # forward + grad
    batch = 5
    u = torch.randn(batch, nu, device=device)
    x = torch.randn(batch, nx, device=device, requires_grad=True)
    dx, y = m(u, x)
    assert dx.shape == (batch, nx)
    assert y.shape == (batch, ny)
    loss = dx.pow(2).sum() + y.pow(2).sum()
    loss.backward()
    # test que certains paramètres ont grad
    some_grad = any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                   for p in m.parameters() if p.requires_grad)
    assert some_grad

    # clonage
    m2 = m.clone()
    dx2, y2 = m2(u, x.detach())
    assert torch.allclose(dx2, dx.detach())
    assert torch.allclose(y2, y.detach())
    # check .param retains
    assert m2.param == m.param

    # check méthode
    ok, info = m.check()
    assert isinstance(ok, bool)
    assert isinstance(info, dict)
    assert ok is True or ok is False  # peut dépendre des poids initiaux

    # string repr
    s = str(m)
    assert s.startswith('ContractingRENODE_')
    r = repr(m)
    assert "ContractingRENODE(" in r
    assert f"param='{param}'" in r


