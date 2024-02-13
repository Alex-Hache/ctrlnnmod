import os
import sys
import torch

from scipy.io import loadmat
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'ctrl_nmod'))

from ctrl_nmod.preprocessing.preprocessing import preprocess_mat_file_dist
from ctrl_nmod.preprocessing.dataset_lib import DataSet
from ctrl_nmod.losses.losses import Mixed_MSELOSS
from ctrl_nmod.models.ssmodels.models_francois import NODE_REN, Sim_discrete
import pandas as pd

_PATH = "data/francois"
_U_LABEL = ["oveHeaPumY_u"]
_TVP_LABEL = ["time",
              "weaSta_reaWeaTDryBul_y",
              "weaSta_reaWeaRelHum_y",
              "weaSta_reaWeaHDirNor_y",
              "weaSta_reaWeaSolAlt_y",
              "InternalGainsCon[1]",
              "InternalGainsLat[1]",
              "InternalGainsRad[1]"]

_TVP_LABEL_EXTENDED = ["weaSta_reaWeaTDryBul_y",
                       "weaSta_reaWeaRelHum_y",
                       "weaSta_reaWeaHDirNor_y",
                       "weaSta_reaWeaSolAlt_y",
                       "Q_occ"]
_Y_LABEL = ["reaTZon_y"]
_FILE_LIST = ["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv", "train_5.csv"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_dataset():
    final_dataset = None
    for file in _FILE_LIST:
        data_frame = pd.read_csv(os.path.join(_PATH, file))
        y = data_frame[_Y_LABEL].values
        u = data_frame[_U_LABEL + _TVP_LABEL].values

        local_dataset = DataSet(u=u, y=y)
        final_dataset = local_dataset if final_dataset is None else local_dataset + final_dataset
    return final_dataset

train_set = get_dataset()
nu = train_set.get_u_dim()
ny = train_set.get_y_dim()
nx = 10
nq = 16

# Emperical Lipschitz estimation
lip= 0.0
for u,y in train_set.iter_exp():
    gy = np.abs(y[1:, :] - y[0:-1, :])
    gu = np.abs(u[1:, :] - u[0:-1, :])
    idx_du_zero = gu != 0
    for k in range(gu.shape[1]):
        t = gy[idx_du_zero[:, k], :]/gu[idx_du_zero[:,k], k]
        if lip <= np.max(t):
            lip = np.max(t)
model = NODE_REN(nx=nx, ny=ny, nu=nu, nq=nq, mode='rl2',
                 alpha=0.99, gamma=100, feedthrough=False)

sim_model = Sim_discrete(model, nb=nx)

best_model, dict_res = sim_model.fit(train_dataset=train_set, test_dataset=train_set, batch_size = 512, 
                                     criterion=Mixed_MSELOSS(alpha=0.0), lr=1e-3, test_freq=1000, epochs=10000, patience=1000)


