
from torch.utils.data import DataLoader
from data.pendulum.load_pendulum import load_pendulum
import numpy as np
import pytest

from ctrlnmod.utils.data import ExperimentsDataset, Experiment


# Length of sequences to consider

@pytest.fixture
def set_seq_len():
    return 10


@pytest.fixture
def order():
    return 2


@pytest.fixture
def x_trainable():
    return True


@pytest.fixture
def experiment(order):
    nx = order
    u_train, y_train, _, _, d_train, d_test, ts = load_pendulum(["data_train_exp2.mat", "data_test_exp2.mat"], scaled=False)
    exp = Experiment(u_train, y_train, ts=ts, nx=nx, x_trainable=True, d=d_train)
    return exp


@pytest.fixture
def exp_yuqi(order):
    nx = order
    u_train_yuqi, y_train_yuqi, _, _, d_train, d_test, ts_yuqi = load_pendulum(["data_train_exp1.mat", "data_test_exp1.mat"], scaled=False)

    exp_yuqi = Experiment(u_train_yuqi, y_train_yuqi, ts=ts_yuqi, nx=nx, x_trainable=True, d=d_train)
    return exp_yuqi


@pytest.fixture
def experiment_nx2(order):
    nx = order
    u_train, y_train, _, _, d_train, d_test, ts = load_pendulum(["data_train_exp2.mat", "data_test_exp2.mat"], scaled=False)
    exp = Experiment(u_train, y_train, ts=ts, nx=2 * nx, x_trainable=True, d=d_train)
    return exp


@pytest.fixture
def experiment_ts2(order):
    nx = order
    u_train, y_train, _, _, d_train, d_test, ts = load_pendulum(["data_train_exp2.mat", "data_test_exp2.mat"], scaled=False)
    exp = Experiment(u_train, y_train, ts=ts + 1e-3, nx=nx, x_trainable=True, d=d_train)
    return exp


@pytest.fixture
def train_set(experiment, set_seq_len):
    train_set = ExperimentsDataset([experiment], seq_len=set_seq_len)

    return train_set


@pytest.fixture
def train_set_exps(experiment, set_seq_len):
    train_set = ExperimentsDataset([experiment, experiment], seq_len=set_seq_len)
    return train_set


@pytest.fixture
def batch_size():
    return 256


def test_experiment_build(order):
    assert isinstance(order, int)
    nx = order
    u_train, y_train, u_test, y_test, d_train, d_test, ts = load_pendulum(["data_train_exp2.mat", "data_test_exp2.mat"], scaled=False)

    # Test if the exceptions are well raised
    with pytest.raises(ValueError):
        Experiment(u_train, y_train, ts=ts, x=np.zeros((nx, u_train.shape[0])), nx=nx, x_trainable=True,
                   d=d_train)

    with pytest.raises(AssertionError):
        Experiment(u_train, np.zeros((y_train.shape[0] + 1, y_train.shape[1])), ts=ts, nx=nx,
                   d=d_train)

    with pytest.raises(ValueError):
        Experiment(u_train, y_train, ts=ts)


def test_dataset_build(set_seq_len, experiment, experiment_nx2, experiment_ts2):

    seq_len = set_seq_len
    train_set = ExperimentsDataset([experiment], seq_len=seq_len)

    with pytest.raises(ValueError):
        ExperimentsDataset([experiment], -2)

    train_set.set_seq_len(-1)
    assert train_set.seq_len == len(experiment.u)

    assert train_set.n_exp == 1

    with pytest.raises(AssertionError):  # Different ts
        train_set_bad_ts = ExperimentsDataset([experiment], seq_len=seq_len)
        train_set_bad_ts.append(experiment_ts2)

    with pytest.raises(AssertionError):  # Different nx
        train_set_bad_nx = ExperimentsDataset([experiment], seq_len=seq_len)

        train_set_bad_nx.append(experiment_nx2)

    train_set.append(experiment)

    assert train_set.n_exp == 2
    assert train_set.n_exp_samples == len(experiment) * 2


def test_dataloader(train_set, batch_size):  # TODO

    data_loader = DataLoader(train_set, batch_size, drop_last=False, shuffle=False)
    assert len(data_loader) == (train_set.n_exp_samples_avl // batch_size) + 1


def test_dataloader_multi_exp(train_set_exps, batch_size, exp_yuqi):

    data_loader = DataLoader(train_set_exps, batch_size, drop_last=False, shuffle=False)
    assert len(data_loader) == (train_set_exps.n_exp_samples_avl // batch_size) + 1  # For the last one

    train_set_exps.append(exp_yuqi)
    data_loader_multi_new = DataLoader(train_set_exps, batch_size, shuffle=True, drop_last=False)

    assert len(data_loader_multi_new) == train_set_exps.n_exp_samples_avl // batch_size + 1


if __name__ == "__main__":

    nx, seq_len, batch_size_var = 2, 10, 256
    u_train, y_train, _, _, ts = load_pendulum(["data_train_exp2.mat", "data_test_exp2.mat"], scaled=False)
    exp = Experiment(u_train, y_train, ts=ts, nx=nx, x_trainable=True)
    u_train_yuqi, y_train_yuqi, _, _, ts_yuqi = load_pendulum(["data_train_exp1.mat", "data_test_exp1.mat"], scaled=False)

    exp_yuqi_var = Experiment(u_train, y_train, ts=ts_yuqi, nx=nx, x_trainable=True)
    train_set_exps_ = ExperimentsDataset([exp, exp_yuqi_var], seq_len=seq_len)
    train_set_exp_ = ExperimentsDataset([exp], seq_len=seq_len)

    data_loader_multi = DataLoader(train_set_exps_, batch_size_var, drop_last=False, shuffle=False)
    data_loader_multi_drop = DataLoader(train_set_exps_, batch_size_var, drop_last=True, shuffle=False)
    data_loader_multi_shuffle = DataLoader(train_set_exps_, batch_size_var, drop_last=True, shuffle=True)

    data_loader_single = DataLoader(train_set_exp_, batch_size_var, drop_last=False, shuffle=False)

    assert len(data_loader_single) == (train_set_exp_.n_exp_samples // batch_size_var) + 1

    for i, batch in enumerate(data_loader_single):
        u_batch, y_batch, x_batch, _ = batch

    # Several experiments
    for i, batch in enumerate(data_loader_multi):
        u_batch, y_batch, x_batch, _ = batch
        print(f" {i} th batch : number of sequences = {u_batch.shape[0]}")

    # Several experiments shuffled
    for i, batch in enumerate(data_loader_multi_shuffle):
        u_batch, y_batch, x_batch, _ = batch
        print(f"Shuffled {i} th batch : number of sequences = {u_batch.shape[0]}")

    for i, batch in enumerate(data_loader_multi_drop):
        u_batch, y_batch, x_batch, _ = batch
        print(f" {i} th batch : number of sequences = {u_batch.shape[0]}")

    train_set_exps_.append(exp_yuqi_var)
    data_loader_multi_new = DataLoader(train_set_exps_, batch_size_var, shuffle=True, drop_last=False)

    for i, batch in enumerate(data_loader_multi_new):
        u_batch, y_batch, x_batch, _ = batch
        print(f"Added experiment shuffled {i} th batch : number of sequences = {u_batch.shape[0]}")

    assert i == train_set_exps_.n_exp_samples // batch_size_var
