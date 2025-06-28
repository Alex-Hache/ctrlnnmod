import torch
torch.set_default_dtype(torch.float64)  # Set default dtype to float64 for precision
import pytest
from ctrlnmod.models.ssmodels.continuous import H2Linear, ExoH2Linear
from torch import Tensor
from ctrlnmod.utils.data import ExperimentsDataset, Experiment, ExperimentsDataModule
from ctrlnmod.integrators.integrators import RK4Simulator
from ctrlnmod.models.ssmodels.continuous.linear.h2 import H2Linear
from ctrlnmod.train.train import train_model, LitNode
from ctrlnmod.losses.losses import MSELoss
from ctrlnmod.regularizations import StateRegularization
from data.pendulum.load_pendulum import load_pendulum
from ctrlnmod.linalg.utils import check_observability, check_controllability
from ctrlnmod.lmis.h2 import H2Cont
import os
from scipy.io import savemat
from torch.optim import Adam

@pytest.fixture
def dims():
    return {"nu": 2, "ny": 3, "nx": 4, "nd": 2, "gamma2": 1.5}


@pytest.fixture
def load_data():

    # Loading data

    u_train, y_train, u_test, y_test, d_train, d_test, ts = load_pendulum(['data_train_exp1.mat', 'data_test_exp1.mat'])
    u_train_yuqi, y_train_yuqi, u_test_yuqi, y_test_yuqi, d_train_yuqi, d_test_yuqi, ts = load_pendulum(['data_train_exp2.mat', 'data_test_exp2.mat'])


    # Length of sequences to consider
    seq_len = 20
    nx = 2

    train_set = ExperimentsDataset([Experiment(u_train, y_train, ts=ts, nx=nx, x_trainable=True, d=d_train)], seq_len)
    train_set.append(Experiment(u_train_yuqi, y_train_yuqi, ts=ts, nx=nx, x_trainable=True, d=d_train_yuqi))
    test_set = ExperimentsDataset([Experiment(u_test, y_test, ts=ts, nx=nx, d=d_test)], seq_len)
    test_set.append(Experiment(u_test_yuqi, y_test_yuqi, ts=ts, nx=nx, x_trainable=False, d=d_test_yuqi))

    nu = 1
    nd = 1
    nx = 2
    ny= 1
    return train_set, test_set, ts, nu, nx, ny, nd


def test_h2linear_forward_shape(dims):
    model = H2Linear(dims["nu"], dims["ny"], dims["nx"], dims["gamma2"])
    u = torch.randn(10, dims["nu"])
    x = torch.randn(10, dims["nx"])
    dx, y = model(u, x)
    assert dx.shape == (10, dims["nx"])
    assert y.shape == (10, dims["ny"])

def test_h2linear_check_and_frame(dims):
    model = H2Linear(dims["nu"], dims["ny"], dims["nx"], dims["gamma2"])
    A, B, C = model._frame()
    assert A.shape == (dims["nx"], dims["nx"])
    assert B.shape == (dims["nx"], dims["nu"])
    assert C.shape == (dims["ny"], dims["nx"])
    ok, gamma = model.check_()
    assert isinstance(ok, bool)
    assert isinstance(gamma, Tensor)

def test_h2linear_clone(dims):
    model = H2Linear(dims["nu"], dims["ny"], dims["nx"], dims["gamma2"])
    clone = model.clone()
    assert isinstance(clone, H2Linear)
    for p1, p2 in zip(model.parameters(), clone.parameters()):
        assert torch.allclose(p1, p2, atol=1e-4)

def test_exoh2linear_forward_shape(dims):
    model = ExoH2Linear(dims["nu"], dims["ny"], dims["nx"], dims["nd"], dims["gamma2"])
    u = torch.randn(8, dims["nu"])
    x = torch.randn(8, dims["nx"])
    d = torch.randn(8, dims["nd"])
    dx, y = model(u, x, d)
    assert dx.shape == (8, dims["nx"])
    assert y.shape == (8, dims["ny"])

def test_exoh2linear_frame_and_shapes(dims):
    model = ExoH2Linear(dims["nu"], dims["ny"], dims["nx"], dims["nd"], dims["gamma2"])
    A, B, C, G = model._frame()
    assert A.shape == (dims["nx"], dims["nx"])
    assert B.shape == (dims["nx"], dims["nu"])
    assert C.shape == (dims["ny"], dims["nx"])
    assert G.shape == (dims["nx"], dims["nd"])

def test_exoh2linear_clone_and_repr(dims):
    model = ExoH2Linear(dims["nu"], dims["ny"], dims["nx"], dims["nd"], dims["gamma2"])
    model2 = model.clone()
    assert isinstance(model2, ExoH2Linear)
    assert str(model).startswith(str(model2)[:10])  # rudimentaire mais utile

def test_exoh2linear_init_weights(dims):
    model = ExoH2Linear(dims["nu"], dims["ny"], dims["nx"], dims["nd"], dims["gamma2"])
    A0 = torch.diag(torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    B0 = torch.ones(dims["nx"], dims["nu"])
    C0 = torch.ones(dims["ny"], dims["nx"])
    G0 = torch.eye(dims["nx"], dims["nd"])
    # Careful when initializing the weights, an ill-conditionned observability gramian can occur
    # if the initial weights are not chosen properly. Please ensure the initial weights leads to an observable system.
    # Please also check
    is_observable = check_observability(A0, C0)
    assert is_observable, "The initial weights do not lead to an observable system."
    model.init_weights_(A0, B0, C0, G0)
    A, B, C, G = model._frame()

    assert torch.dist(A, A0) < 1e-4
    assert torch.dist(B, B0) < 1e-4
    assert torch.dist(C, C0) < 1e-4
    assert torch.dist(G, G0) < 1e-4


def test_h2_linear_end_to_end(dims, load_data):
    train_set, test_set, ts, nu, nx, nd, ny = load_data

    A0 = torch.diag(torch.tensor([-1.0, -4.0]))
    B0 = torch.Tensor([[1], [0]])
    G0 = torch.Tensor([[0], [1]])
    C0 = torch.Tensor([[1, 1]])


    _, gamma2, P = H2Cont.solve(A0, G0, C0)
    model = ExoH2Linear(nu, ny, nx, nd, float(gamma2))
    model.init_weights_(A0, B0, C0, G0)

    sim_model = RK4Simulator(model, ts=torch.Tensor([ts]))
    loss = MSELoss([StateRegularization(model, lambda_state=0.001, update_factor=0.1)])
    val_loss = MSELoss()

    optimizer = Adam(sim_model.parameters(), lr=1e-3)
    logger = None  # Replace with a logger if needed
    # Training options

    batch_size, lr = 512, 1e-4
    epochs, optimizer = 1, optimizer
    exp_module = ExperimentsDataModule(train_set, test_set, batch_size=batch_size, num_workers=0)

    lit_model = LitNode(model=sim_model, criterion=loss, val_criterion=val_loss, lr=lr)

    res = train_model(lit_model=lit_model, data_module=exp_module, logger=logger, epochs=epochs)
