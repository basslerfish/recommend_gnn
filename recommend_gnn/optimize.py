"""
Helper function to optimize hyperparameter search.
"""
import numpy as np
import optuna
import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data

from recommend_gnn.train import train_step


def get_val_loss(
        model: nn.Module,
        loss_fn: nn.Module,
        data: Data,
        i_val: np.ndarray,
) -> float:
    """Basic validation loss."""
    model.eval()
    y_true = torch.squeeze(data.y)
    y_pred = model(data.x, data.edge_index)
    val_loss = loss_fn(y_pred[i_val, :], y_true[i_val])
    val_loss = val_loss.item()
    return val_loss


def train_and_val(
        model: nn.Module,
        data: Data,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        i_train: np.ndarray,
        i_val: np.ndarray,
        n_epochs: int,
        trial: optuna.Trial,
) -> float:
    best_loss = 100.0
    for i_epoch in range(n_epochs):
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            data=data,
            loss_fn=loss_fn,
            i_train=i_train,
        )
        val_loss = get_val_loss(
            model=model,
            i_val=i_val,
            data=data,
            loss_fn=loss_fn,
        )
        if val_loss < best_loss:
            best_loss = val_loss

        # auto-stop
        trial.report(val_loss, i_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_loss
