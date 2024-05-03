from ctrl_nmod.utils.data import ExperimentsDataset, Experiment
from ctrl_nmod.integrators.integrators import RK4Simulator
from ctrl_nmod.models.ssmodels.hinf import L2BoundedLinear
from ctrl_nmod.train.train import SSTrainer
from ctrl_nmod.losses.losses import MixedMSELoss
from data.pendulum.load_pendulum import load_pendulum
from ctrl_nmod.lmis.hinf import HInfCont
from scipy.io import savemat
import os
import torch
torch.set_num_threads(4)


# Loading data

u_train, y_train, u_test, y_test, ts = load_pendulum(['data_train_francois.mat', 'data_test_francois.mat'])
u_train_yuqi, y_train_yuqi, u_test_yuqi, y_test_yuqi, ts = load_pendulum(['data_train_yuqi.mat', 'data_test_yuqi.mat'])


# Length of sequences to consider
seq_len = 20
nx = 2

train_set = ExperimentsDataset([Experiment(u_train, y_train, ts=ts, nx=nx, x_trainable=True)], seq_len)
train_set.append(Experiment(u_train_yuqi, y_train_yuqi, ts=ts, nx=nx, x_trainable=True))
test_set = ExperimentsDataset([Experiment(u_test, y_test, ts=ts, nx=nx)], seq_len)
test_set.append(Experiment(u_test_yuqi, y_test_yuqi, ts=ts, nx=nx, x_trainable=True))

nu, ny, nh = 2, 1, 16
# actF = ReLU()

A0 = -torch.eye(nx)
B0 = torch.Tensor([[0, 1], [1, 0]])
C0 = torch.Tensor([[1, 0]])


_, gamma, P = HInfCont.solve(A0, B0, C0, torch.zeros(ny, nu), alpha=torch.Tensor([-0.9]))
model = L2BoundedLinear(nu, ny, nx, float(gamma))

sim_model = RK4Simulator(model, ts=torch.Tensor([ts]))
loss = MixedMSELoss(alpha=0.0)
val_loss = MixedMSELoss(alpha=0.0)

trainer = SSTrainer(sim_model, loss=loss, val_loss=val_loss)

# Training options

batch_size, lr, keep_best = 512, 1e-3, True
epochs, optimizer = 1000, 'adam'

scheduled = True
step_sched = 0.1
save_path = f'results/try_hinf/{str(trainer)}'
best_model, res = trainer.fit_(train_set=train_set, test_set=test_set,
                               batch_size=batch_size, lr=lr, keep_best=keep_best,
                               save_path=save_path, epochs=epochs, opt=optimizer)

os.makedirs(save_path, exist_ok=True)

savemat(save_path + '/results.mat', res)
