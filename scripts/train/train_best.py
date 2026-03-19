"""
After hyperparameter tuning, train a new model with the best hyperparameters.
"""
import datetime
import os
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from recommend_gnn.model import SageGNN
from recommend_gnn.utils import set_safe_globals
from recommend_gnn.train import train_step, make_splits, write_progress, save_model, evaluate_model

# params
N_EPOCHS = 500
STUDY_NAME = "optimize_gnn_samples10000_prunerhyperband"


def main() -> None:
    # set paths
    current_dir = Path.cwd().parent
    output_dir = current_dir / "results" / "hidden" / "models"
    data_file = current_dir / "data" / "obgn_products_subset10000.pt"
    tb_dir = current_dir / "results" / "hidden" / "tb_runs"
    hp_file = current_dir / "results" / "hidden" / "hyperparams" / "hyperparams.db"

    # find best params in database
    storage = f"sqlite:///{hp_file}"
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=storage,
    )
    best_params = study.best_trial.params
    best_study_loss = study.best_trial.value
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val loss during study: {best_study_loss:.3f}")
    print("Parameters")
    for key, val in best_params.items():
        print(f"\t {key}: {val}")


    # load data
    set_safe_globals()
    data = torch.load(str(data_file))
    n_nodes, n_features = data.x.shape
    n_classes = np.unique(data.y.numpy()).size
    splits = make_splits(n_nodes, 0.2, 0.2)
    best_params["n_out"] = n_classes

    # load GNN
    model = SageGNN(
        n_features=n_features,
        **best_params,
    )
    optimizer = Adam(model.parameters())
    loss_fn = CrossEntropyLoss()

    # prep output
    id_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    id_str = f"{n_nodes}_{id_str}"
    print(f"ID: {id_str}")
    model_dir = output_dir / id_str
    os.makedirs(model_dir)
    log_dir = tb_dir / id_str
    writer = SummaryWriter(str(log_dir))

    # train
    best_loss = 100
    for i_epoch in range(N_EPOCHS):
        train_loss = train_step(
            model=model,
            data=data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            i_train=splits["train"],
        )
        metrics = evaluate_model(
            model=model,
            data=data,
            loss_fn=loss_fn,
            splits={"train": splits["train"], "val": splits["val"]}
        )
        metrics["train_loss"] = train_loss
        write_progress(writer, i_epoch, metrics)
        current_loss = metrics["val_loss"]
        if current_loss < best_loss:
            print(f"New best: {i_epoch=}, {current_loss=:.4f}")
            file_path = model_dir / "best_val.pt"
            save_model(model, best_params, file_path)
            best_loss = current_loss

    # save final
    file_path = model_dir / "final.pt"
    save_model(model, best_params, file_path)


if __name__ == "__main__":
    main()