import torch
torch.set_default_dtype(torch.float64)
from ctrlnmod.models.ssmodels.continuous import FLNSSM, ExoSSLinear
from ctrlnmod.losses import MSELoss
from data.pendulum.load_pendulum import load_pendulum
from ctrlnmod.utils.data import Experiment, ExperimentsDataModule, ExperimentsDataset
from ctrlnmod.train.train import train_model, LitNode
from ctrlnmod.integrators import RK45Simulator
from lightning.pytorch.loggers import TensorBoardLogger
import os 

def set_hyperparameters():
    return {
        'seq_len' : 20,
        'batch_size': 512,
        'nx' : 2,
        'lr' : 1e-2,
        'epochs' : 2,
        'lambda_logdet' : 1e-2

    }

def load_data(hyperparams):

    u_train, y_train, u_test, y_test, d_train, d_test, ts = load_pendulum(['data_train_exp1.mat', 'data_test_exp1.mat'])
    exp1_train = Experiment(u_train, y_train, ts, nx=hyperparams['nx'], x_trainable=True, d=d_train)
    exp1_test = Experiment(u_test, y_test, ts, nx=hyperparams['nx'], x_trainable=False, d=d_test)

    u_train2, y_train2, u_test2, y_test2, d_train2, d_test2, ts2 = load_pendulum(['data_train_exp2.mat', 'data_test_exp2.mat'])
    exp2_train = Experiment(u_train2, y_train2, ts2, nx=hyperparams['nx'], x_trainable=True, d=d_train2)
    exp2_test = Experiment(u_test2, y_test2, ts2, nx=hyperparams['nx'], x_trainable=False, d=d_test2)

    train_set = ExperimentsDataset([exp1_train, exp2_train], seq_len=hyperparams['seq_len'])
    test_set = ExperimentsDataset([exp1_test, exp2_test], seq_len=u_test.shape[0])

    return ExperimentsDataModule(train_set=train_set, val_set=test_set,batch_size=hyperparams['batch_size'])

def set_dims():
    return {
        'nu' : 1,
        'ny' : 1,
        'nd': 1,
        'nx': 2,
        'hidden_layers' : [32]
    }


def main():
    hyperparams = set_hyperparameters()
    dims = set_dims()
    data_module = load_data(hyperparams)
    nu, ny, nx, nd, hidden_layers = dims['nu'], dims['ny'], dims['nx'], dims['nd'], dims['hidden_layers']


    linear_model = ExoSSLinear(ny, ny, nx, nd, alpha=1e-2)

    model = FLNSSM(nu, ny, nx, linear_model, 'output_feedback', nd,
                    hidden_layers)
    
    sim_model = RK45Simulator(model, data_module.ts)

    # Custom logging function
    lit_model = LitNode(sim_model,MSELoss(), MSELoss(), lr=hyperparams['lr'])

    # logger
    res_dir = 'results/base_pendulum'
    os.makedirs(res_dir, exist_ok=True)

    logger = TensorBoardLogger(res_dir,'base_pendulum')
    trainer = train_model(lit_model, data_module, logger, epochs=hyperparams['epochs'])

if __name__ == '__main__':
    main()