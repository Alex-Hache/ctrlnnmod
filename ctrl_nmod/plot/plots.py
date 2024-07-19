import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Union, Optional
from matplotlib.figure import Figure

# Type alias for arrays that can be either NumPy arrays or PyTorch tensors
ArrayLike = Union[np.ndarray, torch.Tensor]


def to_numpy(array: Union[ArrayLike, List[float]]) -> np.ndarray:
    """Convert input to a NumPy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def plot_yTrue_vs_ySim(yTrue: ArrayLike, ySim: ArrayLike, save_path: str) -> Figure:
    """
    Plot true values against simulated values.

    Args:
        yTrue (ArrayLike): An array or tensor containing true data of shape (N_samples, N_channels).
        ySim (ArrayLike): An array or tensor containing simulated output of shape (N_samples, N_channels).
        save_path (str): Path to save the generated figure.

    Returns:
        Figure: The matplotlib Figure object containing the plot.

    Raises:
        ValueError: If the shapes of yTrue and ySim do not match.
    """
    yTrue_np = to_numpy(yTrue)
    ySim_np = to_numpy(ySim)

    if yTrue_np.shape != ySim_np.shape:
        raise ValueError("The shapes of yTrue and ySim must match.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yTrue_np, label='True')
    ax.plot(ySim_np, label='Simulated')

    ax.set_xlabel('Samples')
    ax.set_ylabel('Values')
    ax.set_title('True vs Simulated Values')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return fig


def plot_yTrue_vs_error(yTrue: ArrayLike, ySim: ArrayLike, save_path: str) -> Figure:
    """
    Plot true values against the error (difference between true and simulated values).

    Args:
        yTrue (ArrayLike): An array or tensor containing true data of shape (N_samples, N_channels).
        ySim (ArrayLike): An array or tensor containing simulated output of shape (N_samples, N_channels).
        save_path (str): Path to save the generated figure.

    Returns:
        Figure: The matplotlib Figure object containing the plot.

    Raises:
        ValueError: If the shapes of yTrue and ySim do not match.
    """
    yTrue_np = to_numpy(yTrue)
    ySim_np = to_numpy(ySim)

    if yTrue_np.shape != ySim_np.shape:
        raise ValueError("The shapes of yTrue and ySim must match.")

    fig, ax = plt.subplots(figsize=(10, 6))
    err_sim = yTrue_np - ySim_np
    ax.plot(yTrue_np, label='True')
    ax.plot(err_sim, label='Error')

    ax.set_xlabel('Samples')
    ax.set_ylabel('Values')
    ax.set_title('True Values vs Error')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return fig


def plot_losses(
    train_losses: Union[List[float], ArrayLike],
    val_losses: Union[List[float], ArrayLike],
    use_log_scale: bool = True,
    save_path: Optional[str] = None,
    title: str = "Training and Validation Losses",
    xlabel: str = "Epochs",
    ylabel: str = "Loss",
    figsize: tuple = (10, 6)
) -> Figure:
    """
    Plot training and validation losses.

    Args:
        train_losses (Union[List[float], ArrayLike]): List, array, or tensor of training loss values.
        val_losses (Union[List[float], ArrayLike]): List, array, or tensor of validation loss values.
        use_log_scale (bool, optional): If True, display losses on a log10 scale. Defaults to True.
        save_path (Optional[str], optional): Path to save the generated figure. If None, the figure is not saved. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Training and Validation Losses".
        xlabel (str, optional): Label for x-axis. Defaults to "Epochs".
        ylabel (str, optional): Label for y-axis. Defaults to "Loss".
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (10, 6).

    Returns:
        Figure: The matplotlib Figure object containing the plot.

    Raises:
        ValueError: If the lengths of train_losses and val_losses are not equal.
    """
    train_losses_np = to_numpy(train_losses)
    val_losses_np = to_numpy(val_losses)

    if len(train_losses_np) != len(val_losses_np):
        raise ValueError(
            "The lengths of train_losses and val_losses must be equal.")

    if use_log_scale:
        train_losses_np = np.log10(train_losses_np)
        val_losses_np = np.log10(val_losses_np)
        ylabel = f"Log {ylabel}"

    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_losses_np) + 1)

    ax.plot(epochs, train_losses_np, label='Training Loss', marker='o')
    ax.plot(epochs, val_losses_np, label='Validation Loss', marker='s')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close(fig)

    return fig


# Example usage:
# import torch
#
# # NumPy arrays
# yTrue_np = np.random.rand(100, 3)
# ySim_np = np.random.rand(100, 3)
# plot_yTrue_vs_ySim(yTrue_np, ySim_np, 'true_vs_sim_np.png')
# plot_yTrue_vs_error(yTrue_np, ySim_np, 'true_vs_error_np.png')
#
# # PyTorch tensors
# yTrue_torch = torch.rand(100, 3)
# ySim_torch = torch.rand(100, 3)
# plot_yTrue_vs_ySim(yTrue_torch, ySim_torch, 'true_vs_sim_torch.png')
# plot_yTrue_vs_error(yTrue_torch, ySim_torch, 'true_vs_error_torch.png')
#
# # Losses can be lists, NumPy arrays, or PyTorch tensors
# train_losses = [1.5, 1.2, 1.0, 0.8, 0.7]
# val_losses = torch.tensor([1.6, 1.3, 1.1, 0.9, 0.85])
# plot_losses(train_losses, val_losses, save_path='losses.png')
