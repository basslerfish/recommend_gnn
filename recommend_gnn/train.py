"""
Functions related to training the GNN.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from sklearn.metrics import balanced_accuracy_score, f1_score

def train(
        model: nn.Module,
        optimizer: optim.Optimizer,
        data: Data,
        loss_fn: nn.Module,
        i_train: torch.Tensor | np.ndarray,
        y_true: torch.Tensor,
) -> float:
    """Single training iteration."""
    model.train()
    optimizer.zero_grad()
    y_pred = model(data.x, data.edge_index)
    loss = loss_fn(y_pred[i_train, :], y_true[i_train])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(
        model: nn.Module,
        data: Data,
        y_true: torch.Tensor,
        loss_fn: nn.Module,
        i_train: torch.Tensor | np.ndarray,
        i_test: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()

    # test loss
    y_pred = model(data.x, data.edge_index)
    test_loss = loss_fn(y_pred[i_test, :], y_true[i_test])

    # other metrics (sklearn-based)
    y_pred = y_pred.detach().numpy()
    pred_index = np.argmax(y_pred, axis=1)
    y_true_np = y_true.numpy()
    metrics = {
        "test_loss": test_loss.item(),
    }
    selections = {
        "train": i_train,
        "test": i_test,
    }
    functions = {
        "accuracy": balanced_accuracy_score,
        "f1": partial(f1_score, average="weighted"),
    }
    for sel_name, index_vec in selections.items():
        for score_name, func in functions.items():
            score = func(y_true_np[index_vec], pred_index[index_vec])
            metrics[f"{sel_name}_{score_name}"] = score
    return metrics


def write_progress(
        writer: SummaryWriter,
        i_epoch: int,
        train_loss: float,
        metrics: dict,
) -> None:
    writer.add_scalar("train_loss", train_loss, i_epoch)
    for key, val in metrics.items():
        writer.add_scalar(key, val, i_epoch)