"""
Train a GNN to predict product category from bag-of-words features and co-purchase information.
Manual setting of hyperparameters required in this version.

Saves training information with tensorboard.
"""
import datetime
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from recommend_gnn.utils import set_safe_globals
from recommend_gnn.model import SageGNN
from recommend_gnn.train import train_step, evaluate_model, write_progress, make_splits, save_model


# hyperparams
N_HIDDEN = 128
DEPTH = 2
N_EPOCHS = 500
VAL_FRAC = 0.2
TEST_FRAC = 0.2
SAGE_AGGREGATE = "mean"
SAGE_PROJECT = False
JK_AGGREGATE = "cat"
DROPOUT_RATE = 0.5
DROPOUT_LAST = True


def main() -> None:
    # set paths
    current_dir = Path.cwd().parent
    output_dir = current_dir / "results" / "hidden" / "models"
    data_file = current_dir / "data" / "obgn_products_subset10000.pt"
    tb_dir = current_dir / "results" / "hidden" / "tb_runs"

    # load data
    set_safe_globals()
    data = torch.load(data_file)
    n_nodes, n_features = data.x.shape
    print(f"Nodes (products): {n_nodes:,}")
    print(f"Edges: {data.edge_index.shape[1]:,}")
    print(f"Features per node: {n_features}")
    n_classes = np.unique(data.y).size
    print(f"Product classes: {n_classes}")

    # make train mask
    splits = make_splits(n_nodes, val_frac=VAL_FRAC, test_frac=TEST_FRAC)

    # make model
    model = SageGNN(
        n_features=n_features,
        n_hidden=N_HIDDEN,
        depth=DEPTH,
        sage_aggregate=SAGE_AGGREGATE,
        jk_aggregate=JK_AGGREGATE,
        dropout_rate=0.5,
        n_out=n_classes,
        sage_project=SAGE_PROJECT,
        dropout_last=DROPOUT_LAST,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # model output
    id_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    id_str = f"{n_nodes}_{id_str}"
    print(f"ID: {id_str}")
    model_dir = output_dir / id_str
    os.makedirs(model_dir)

    # prepare tensorboard
    log_dir = tb_dir / id_str
    writer = SummaryWriter(str(log_dir))
    hyper_params = {
        "n_hidden": N_HIDDEN,
        "depth": DEPTH,
        "sage_aggregate": SAGE_AGGREGATE,
        "sage_project": SAGE_PROJECT,
        "jk_aggregate": JK_AGGREGATE,
        "val_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
    }

    # train
    best_loss = 100
    for i_epoch in range(N_EPOCHS):
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            data=data,
            i_train=splits["train"],
            loss_fn=loss_fn,
        )
        metrics = evaluate_model(
            model=model,
            data=data,
            splits={"train": splits["train"], "val": splits["val"]},
            loss_fn=loss_fn,
        )
        metrics["train_loss"] = train_loss
        write_progress(
            writer=writer,
            i_epoch=i_epoch,
            metrics=metrics,
        )
        print(f"{i_epoch}: {train_loss=:.3f}, val_loss={metrics['val_loss']:.3f}")
        if metrics["val_loss"] < best_loss:
            print("Best validation loss yet.")
            file_path = model_dir / "best_val.pt"
            save_model(model, hyper_params, file_path, i_epoch=i_epoch, **metrics)
            best_loss = metrics["val_loss"]

    # final evaluation
    with torch.no_grad():
        test_metrics = evaluate_model(
            model=model,
            splits={"test": splits["test"]},
            data=data,
            loss_fn=loss_fn,
        )
        file_path = model_dir / "final.pt"
        save_model(model, hyper_params, file_path, **test_metrics)
        print(f"Test accuracy: {test_metrics['test_accuracy']:.3f}")
        writer.add_hparams(
            hparam_dict=hyper_params,
            metric_dict=test_metrics,
        )

if __name__ == "__main__":
    main()