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


def _plot_base(
    x_data: np.ndarray,
    y_data: List[np.ndarray],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    figsize=(10, 6),
) -> Figure:
    """Helper function to create and save a plot with multiple lines."""
    fig, ax = plt.subplots(figsize=figsize)
    for y, label in zip(y_data, labels):
        ax.plot(y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return fig


def plot_y_true_vs_y_sim(
    y_true: ArrayLike, y_sim: ArrayLike, save_path: str
) -> Figure:
    """
    Plot true values against simulated values.

    Args:
        y_true (ArrayLike): True data of shape (N_samples, N_channels).
        y_sim (ArrayLike): Simulated output of shape (N_samples, N_channels).
        save_path (str): Path to save the generated figure.

    Returns:
        Figure: Matplotlib Figure object containing the plot.

    Raises:
        ValueError: If shapes of y_true and y_sim do not match.
    """
    y_true_np = to_numpy(y_true)
    y_sim_np = to_numpy(y_sim)

    if y_true_np.shape != y_sim_np.shape:
        raise ValueError("Shapes of y_true and y_sim must match.")

    return _plot_base(
        x_data=np.arange(y_true_np.shape[0]),
        y_data=[y_true_np, y_sim_np],
        labels=["True", "Simulated"],
        xlabel="Samples",
        ylabel="Values",
        title="True vs Simulated Values",
        save_path=save_path,
    )


def plot_y_true_vs_error(
    y_true: ArrayLike, y_sim: ArrayLike, save_path: str
) -> Figure:
    """
    Plot true values against the error (difference between true and simulated values).

    Args:
        y_true (ArrayLike): True data of shape (N_samples, N_channels).
        y_sim (ArrayLike): Simulated output of shape (N_samples, N_channels).
        save_path (str): Path to save the generated figure.

    Returns:
        Figure: Matplotlib Figure object containing the plot.

    Raises:
        ValueError: If shapes of y_true and y_sim do not match.
    """
    y_true_np = to_numpy(y_true)
    y_sim_np = to_numpy(y_sim)

    if y_true_np.shape != y_sim_np.shape:
        raise ValueError("Shapes of y_true and y_sim must match.")

    error = y_true_np - y_sim_np
    return _plot_base(
        x_data=np.arange(y_true_np.shape[0]),
        y_data=[y_true_np, error],
        labels=["True", "Error"],
        xlabel="Samples",
        ylabel="Values",
        title="True Values vs Error",
        save_path=save_path,
    )


def plot_losses(
    train_losses: Union[List[float], ArrayLike],
    val_losses: Union[List[float], ArrayLike],
    use_log_scale: bool = True,
    save_path: Optional[str] = None,
    title: str = "Training and Validation Losses",
    xlabel: str = "Epochs",
    ylabel: str = "Loss",
    figsize: tuple = (10, 6),
) -> Figure:
    """
    Plot training and validation losses.

    Args:
        train_losses (Union[List[float], ArrayLike]): Training loss values.
        val_losses (Union[List[float], ArrayLike]): Validation loss values.
        use_log_scale (bool, optional): Use log10 scale for y-axis. Defaults to True.
        save_path (Optional[str], optional): Path to save figure. If None, does not save.
        title (str, optional): Plot title. Defaults to "Training and Validation Losses".
        xlabel (str, optional): X-axis label. Defaults to "Epochs".
        ylabel (str, optional): Y-axis label. Defaults to "Loss".
        figsize (tuple, optional): Figure size. Defaults to (10, 6).

    Returns:
        Figure: Matplotlib Figure object containing the plot.

    Raises:
        ValueError: If lengths of train_losses and val_losses differ.
    """
    train_np = to_numpy(train_losses)
    val_np = to_numpy(val_losses)

    if len(train_np) != len(val_np):
        raise ValueError("Lengths of train_losses and val_losses must be equal.")

    if use_log_scale:
        train_np = np.log10(train_np)
        val_np = np.log10(val_np)
        ylabel = f"Log {ylabel}"

    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_np) + 1)

    ax.plot(epochs, train_np, label="Training Loss", marker="o")
    ax.plot(epochs, val_np, label="Validation Loss", marker="s")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    return fig


# Example usage:
# import torch
#
# y_true_np = np.random.rand(100, 3)
# y_sim_np = np.random.rand(100, 3)
# plot_y_true_vs_y_sim(y_true_np, y_sim_np, 'true_vs_sim_np.png')
# plot_y_true_vs_error(y_true_np, y_sim_np, 'true_vs_error_np.png')
#
# y_true_torch = torch.rand(100, 3)
# y_sim_torch = torch.rand(100, 3)
# plot_y_true_vs_y_sim(y_true_torch, y_sim_torch, 'true_vs_sim_torch.png')
# plot_y_true_vs_error(y_true_torch, y_sim_torch, 'true_vs_error_torch.png')
#
# train_losses = [1.5, 1.2, 1.0, 0.8, 0.7]
# val_losses = torch.tensor([1.6, 1.3, 1.1, 0.9, 0.85])
# plot_losses(train_losses, val_losses, save_path='losses.png')
