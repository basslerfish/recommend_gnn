"""
Functions related to training the GNN.
"""
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def make_splits(n_nodes: int, val_frac: float, test_frac: float) -> dict:
    """
    Make train, val and test splits.
    """
    n_val = int(n_nodes * val_frac)
    n_test = int(n_nodes * test_frac)
    n_train = n_nodes - (n_val + n_test)

    i_all_shuffled = np.random.permutation(n_nodes)

    i_train = i_all_shuffled[:n_train]
    i_val = i_all_shuffled[n_train:n_train+n_val]
    i_test = i_all_shuffled[n_train+n_val:]
    selections = {
        "train": i_train,
        "val": i_val,
        "test": i_test,
    }
    return selections


def train_step(
        model: nn.Module,
        optimizer: optim.Optimizer,
        data: Data,
        loss_fn: nn.Module,
        i_train: torch.Tensor | np.ndarray,
) -> float:
    """Single training iteration."""
    y_true = torch.squeeze(data.y)
    model.train()
    optimizer.zero_grad()
    y_pred = model(data.x, data.edge_index)
    loss = loss_fn(y_pred[i_train, :], y_true[i_train])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_model(
        model: nn.Module,
        data: Data,
        loss_fn: nn.Module,
        splits: dict[int, np.ndarray],
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()
    y_pred = model(data.x, data.edge_index)
    y_true = torch.squeeze(data.y)
    metrics = {}

    # loss
    for split_name, i_split in splits.items():
        sel_loss = loss_fn(y_pred[i_split, :], y_true[i_split])
        sel_loss = sel_loss.item()
        metrics[f"{split_name}_loss"] = sel_loss

    # other metrics (sklearn-based)
    y_pred = y_pred.detach().numpy()
    pred_index = np.argmax(y_pred, axis=1)
    y_true_np = y_true.numpy()
    functions = {
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
        "f1": partial(f1_score, average="weighted"),
    }
    for split_name, i_split in splits.items():
        for score_name, func in functions.items():
            score = func(y_true_np[i_split], pred_index[i_split])
            metrics[f"{split_name}_{score_name}"] = score
    return metrics


def write_progress(
        writer: SummaryWriter,
        i_epoch: int,
        metrics: dict,
) -> None:
    """Write tensorboard data."""
    for key, val in metrics.items():
        writer.add_scalar(key, val, i_epoch)


def save_model(model: nn.Module, hyper_params: dict, file_path: Path, **kwargs) -> None:
    """Save model"""
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_params": hyper_params,
    }
    if kwargs:
        checkpoint.update(kwargs)
    torch.save(checkpoint, str(file_path))
