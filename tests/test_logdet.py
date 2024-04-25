import torch
import sys
sys.path.append("../ctrl_nmod")
sys.path.append("../data")
from data.pendulum.load_pendulum import load_pendulum
import pytest
from ctrl_nmod.utils.data import ExperimentsDataset, Experiment
from ctrl_nmod.utils.misc import find_module
from ctrl_nmod.integrators.integrators import RK4Simulator
from ctrl_nmod.models.ssmodels.grnssm import Grnssm
from ctrl_nmod.models.ssmodels.linear import NnLinear
from ctrl_nmod.losses.losses import MixedMSELoss
from ctrl_nmod.losses.regularizations import LMILogdet
from ctrl_nmod.lmis.hinf import HInfCont

torch.set_num_threads(4)


@pytest.fixture
def build_model():

    # Loading data

    u_train, y_train, u_test, y_test, ts = load_pendulum(
        ["data_train_francois.mat", "data_test_francois.mat"], scaled=False)
    u_train_yuqi, y_train_yuqi, u_test_yuqi, y_test_yuqi, ts_yuqi = load_pendulum(
        ["data_train_yuqi.mat", "data_test_yuqi.mat"], scaled=False)

    # Length of sequences to consider
    seq_len, nx = 20, 2

    train_set = ExperimentsDataset(
        [Experiment(u_train, y_train, ts=ts, nx=nx, x_trainable=True)], seq_len)
    test_set = ExperimentsDataset(
        [Experiment(u_test, y_test, ts=ts, nx=nx)], seq_len)

    # Adding another eperiment
    train_set.append(Experiment(u_train_yuqi, y_train_yuqi, nx=nx,
                     ts=ts_yuqi, x_trainable=True))
    test_set.append(Experiment(u_test_yuqi, y_test_yuqi, ts=ts_yuqi, nx=nx))

    nu, ny, nh = 2, 1, 16
    # actF = ReLU()
    model = Grnssm(nu, ny, nx, nh)
    # A0, B0, C0, D0 = findBLA(u_train, y_train, nx, ts, model_type='continuous')
    A0 = -torch.eye(nx)
    B0 = model.linmod.B.weight
    C0 = model.linmod.C.weight
    model.init_weights_(A0, B0, C0)
    sim_model = RK4Simulator(model, ts=torch.Tensor([ts]))
    return sim_model


@pytest.fixture
def alpha1():
    return 1.0


@pytest.fixture
def alpha0():
    return 0.0


@pytest.fixture
def scaler():
    return 10


@pytest.fixture
def mu0():
    return 0.1


def test_build_reg_mse1(alpha1, scaler):
    loss = MixedMSELoss(alpha=alpha1, scale=scaler)

    assert loss.alpha == alpha1

    loss.update()

    assert loss.alpha == alpha1 * scaler


def test_build_logdet(build_model, mu0, alpha0):
    sim_model = build_model
    linmod = find_module(sim_model, NnLinear)

    hinf = HInfCont(linmod.A.weight, linmod.B.weight, linmod.C.weight)  # type: ignore
    criterion = MixedMSELoss(alpha0)

    criterion.append(LMILogdet(hinf, mu0))

    assert len(criterion.regs) == 2

    weights = criterion.get_weights()
    assert weights[0] == alpha0
    assert weights[1] == mu0

    criterion.update()
    weights = criterion.get_weights()
    scalers = criterion.get_scalers()
    assert weights[0] == alpha0 * scalers[0]
    assert weights[1] == mu0 * scalers[1]


def test_multi_lmi(build_model, mu0, alpha0):
    sim_model = build_model
    linmod = find_module(sim_model, NnLinear)

    hinf = HInfCont(linmod.A.weight, linmod.B.weight, linmod.C.weight)  # type: ignore
    criterion = MixedMSELoss(alpha0)

    criterion.append(LMILogdet(hinf, mu0))

    criterion.append(LMILogdet(hinf, mu0, scale=0.1))
    assert len(criterion.regs) == 3

    criterion.update()
    weights = criterion.get_weights()
    scalers = criterion.get_scalers()
    assert weights[0] == alpha0 * scalers[0]
    assert weights[1] == mu0 * scalers[1]
    assert weights[2] == mu0 * scalers[2]
    criterion.pop(2)
    assert len(criterion.regs) == 2
