from scipy.io import loadmat
import numpy as np
from typing import List, Tuple


def load_pendulum(files: List[str], scaled: bool = False) -> Tuple:

    train_data = loadmat('data/pendulum/' + files[0])
    test_data = loadmat('data/pendulum/' + files[1])

    u_train, y_train, d_train, fs = train_data['uTot'], train_data['yTot'], train_data['pTot'], train_data['fs']
    u_test, y_test, d_test = test_data['uTot'], test_data['yTot'], test_data['pTot']

    if u_train.shape[0] < u_train.shape[1]:
        u_train = np.transpose(u_train)
    if d_train.shape[0] < d_train.shape[1]:
        d_train = np.transpose(d_test)


    # Scale data
    u_max = np.max(u_train, axis=0)
    y_max = np.max(y_train, axis=0)

    ts = 1 / fs[0][0]
    if scaled:
        u_train_scl = u_train / u_max
        y_train_scl = y_train / y_max
        d_train_scl = d_train / y_max
        u_test_scl = u_test / u_max
        y_test_scl = y_test / y_max
        d_test_scl = d_test / y_max

        return u_train_scl, y_train_scl, u_test_scl, y_test_scl, d_train_scl, d_test_scl, ts
    else:
        return u_train, y_train, u_test, y_test, d_train, d_test, ts
