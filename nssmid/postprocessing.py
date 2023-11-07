import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

def plot_yTrue_vs_ySim(yTrue, ySim):
    """
        params : 
            * y_true : a numpy array that contains data of N_samples x N_channels
            * y_pred : a numpy array that contains simulated output N_samples x N_channels
    
    """
    fig, ax = plt.subplots()
    ax.plot(yTrue)
    ax.plot(ySim)
    plt.show()
    return fig

def plot_yTrue_vs_error(yTrue, ySim):
    """
        params : 
            * y_true : a numpy array that contains data of N_samples x N_channels
            * y_pred : a numpy array that contains simulated output N_samples x N_channels
    
    """
    fig, ax = plt.subplots()
    err_sim = yTrue - ySim
    ax.plot(yTrue)
    ax.plot(err_sim)
    plt.show()
    return fig

def plot_losses(v_loss, v_val_loss, b_log = True):

    l = np.array(v_loss)
    vl = np.array(v_val_loss)
    if b_log:
        l = np.log10(l)
        vl = np.log10(vl)
    fig = plt.figure()
    plt.plot(l, label = 'training loss')
    plt.plot(vl , label = 'test loss')

    fig.legend(loc = 'upper right')
    plt.show()
    return fig

def save_weights(weights, biases, strName):

    
    mDict = {'weights' : weights, 'biases' : biases}
    savemat(strName, mDict)

def setSavingName(model, nh, n_layers, n_seq, n_batch, lr, optimizer, n_iter, gamma):

    strMatFileName = f"Res_{model}_{nh}_{n_layers}_{n_seq}_{n_batch}_{lr}_{optimizer}_{n_iter}epch_{gamma :.3e}.mat"
    strLossFigName = f"Loss_{model}_{nh}_{n_layers}_{n_seq}_{n_batch}_{lr}_{optimizer}_{n_iter}epch_{gamma :.3e}.png"
    strSimFigName = f"Sim_{model}_{nh}_{n_layers}_{n_seq}_{n_batch}_{lr}_{optimizer}_{n_iter}epch_{gamma :.3e}.png"

    return strMatFileName, strLossFigName, strSimFigName

