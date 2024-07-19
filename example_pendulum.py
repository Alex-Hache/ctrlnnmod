from ctrl_nmod.utils.data import ExperimentsDataset, Experiment
from ctrl_nmod.integrators.integrators import RK4Simulator
from ctrl_nmod.models.ssmodels.grnssm import Grnssm
from ctrl_nmod.train.train import SSTrainer
from ctrl_nmod.losses.losses import MSELoss, NMSELoss
from ctrl_nmod.regularizations.regularizations import StateRegularization
from data.pendulum.load_pendulum import load_pendulum
from ctrl_nmod.plot.plots import plot_yTrue_vs_error
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
# actF = 'relu'
model = Grnssm(nu, ny, nx, nh)
# A0, B0, C0, D0 = findBLA(u_train, y_train, nx, float(1/fs[0]), model_type='continuous')
A0 = -torch.eye(nx)
B0 = torch.Tensor([[0, 1], [1, 0]])
C0 = torch.Tensor([[1, 0]])

model.init_weights_(A0, B0, C0)
sim_model = RK4Simulator(model, ts=torch.Tensor([ts]))
loss = MSELoss([StateRegularization(model, 0.0, 0.0, updatable=False, verbose=True)])
val_loss = NMSELoss()

trainer = SSTrainer(sim_model, loss=loss, val_loss=val_loss)

# Training options

batch_size, lr, keep_best = 256, 1e-3, True
epochs, optimizer = 10, 'adam'

scheduled = True
step_sched = 0.1
save_path = f'results/try_1_no_scl/{str(trainer)}'
best_model, res = trainer.fit_(train_set=train_set, test_set=test_set,
                               batch_size=batch_size, lr=lr, keep_best=keep_best,
                               save_path=save_path, epochs=epochs, opt=optimizer)


# Best model simulation on test set
x_sim, y_sim = best_model.simulate(test_set.experiments[0].u, torch.zeros(nx))

plot_yTrue_vs_error(test_set.experiments[0].y, y_sim, save_path + 'test_exp_1_sim.png')

os.makedirs(save_path, exist_ok=True)

savemat(save_path + '/results.mat', res)

torch.save(best_model, save_path + '/model.pkl')
