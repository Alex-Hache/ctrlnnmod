import torch
import numpy as np


def preprocess_mat_file(dMat: dict, nx: int, idxMax: int = 100000):
    fs = dMat['fs']
    ts = 1 / fs[0][0]
    u = dMat['uTot']
    y = dMat['yTot']

    if u.shape[0] < u.shape[1]:  # number of samples is on dimension 1
        u = u.T
    if y.shape[0] < y.shape[1]:
        y = y.T

    u_torch = torch.from_numpy(u[:idxMax, :]).to(dtype=torch.float32)
    y_torch = torch.from_numpy(y[:idxMax, :]).to(dtype=torch.float32)

    try:  # Do we have disturbances ?
        d = dMat['pTot']
        if d.shape[0] < d.shape[1]:
            d = d.T
        d_torch = torch.from_numpy(d[:idxMax, :]).to(dtype=torch.float32)
        u_torch = torch.cat((u_torch, d_torch), dim=1)
    except:
        u_torch = torch.from_numpy(u[:idxMax, :]).to(dtype=torch.float32)

    try:  # Do we have state measurements ?
        x = dMat['xTot']
        x = np.reshape(x, (max(x.shape), min(x.shape)))
        x_torch = torch.from_numpy(x[:idxMax, :]).to(dtype=torch.float32)
    except:
        x_torch = torch.zeros(
            (y.shape[0], nx), dtype=torch.float32, requires_grad=True)

    y_torch_dot = (y_torch[1:, :] - y_torch[0:-1, :]) / ts
    y_torch_dot = torch.cat([torch.zeros((1, y.shape[1])), y_torch_dot])

    return u_torch, y_torch, x_torch, y_torch_dot, ts


def preprocess_mat_file_dist(dMat: dict, nx: int, idxMax: int = 100000):
    fs = dMat['fs']
    ts = 1 / fs[0][0]
    u = dMat['uTot']
    y = dMat['yTot']
    d = dMat['pTot']
    if u.shape[0] < u.shape[1]:  # number of samples is on dimension 1
        u = u.T
    if y.shape[0] < y.shape[1]:
        y = y.T
    if d.shape[0] < d.shape[1]:
        d = d.T

    u_torch = torch.from_numpy(u[:idxMax, :]).to(dtype=torch.float32)
    y_torch = torch.from_numpy(y[:idxMax, :]).to(dtype=torch.float32)
    d_torch = torch.from_numpy(d[:idxMax, :]).to(dtype=torch.float32)

    try:
        x = dMat['xTot']
        x = np.reshape(x, (max(x.shape), min(x.shape)))
        x_torch = torch.from_numpy(x[:idxMax, :]).to(dtype=torch.float32)
    except:
        x_torch = torch.zeros(
            (y.shape[0], nx), dtype=torch.float32, requires_grad=True)

    u_torch = torch.cat((u_torch, d_torch), dim=1)
    y_torch_dot = (y_torch[1:, :] - y_torch[0:-1, :]) / ts
    y_torch_dot = torch.cat([torch.zeros((1, y.shape[1])), y_torch_dot])

    return u_torch, y_torch, x_torch, y_torch_dot, ts
