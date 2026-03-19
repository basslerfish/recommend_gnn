"""
Use Optuna for Bayesian optimization of model hyperparameters.
"""
from pathlib import Path

import optuna
import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from recommend_gnn.optimize import train_and_val
from recommend_gnn.model import SageGNN
from recommend_gnn.utils import  set_safe_globals
from recommend_gnn.train import make_splits


# params
N_TRIALS = 100
N_STARTUP_TRIALS = 10
DATA_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/obgn_products_subset10000.pt")
OUTPUT_DIR = Path("/Users/mathis/Code/github/recommend_gnn/results/hyperparams")
VAL_FRAC = 0.2
TEST_FRAC = 0.2
N_EPOCHS = 200
N_JOBS = 4

# load data
print("---Loading data---")
set_safe_globals()
data = torch.load(DATA_FILE)
n_nodes, n_features = data.x.shape
n_classes = np.unique(data.y).size
splits = make_splits(n_nodes, val_frac=VAL_FRAC, test_frac=TEST_FRAC)


# funcs
def objective(trial: optuna.trial.Trial) -> float:
    """Minimize val_loss across a range of hyperparameters."""
    n_hidden = trial.suggest_int("n_hidden", 32, 512, step=32)
    depth = trial.suggest_int("depth", 2, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.8, step=0.1)
    sage_aggregate = trial.suggest_categorical("sage_aggregate", ["mean", "max"])
    jk_aggregate = trial.suggest_categorical("jk_aggregate", ["max", "cat"])
    model = SageGNN(
        n_features=n_features,
        n_hidden=n_hidden,
        depth=depth,
        dropout_rate=dropout_rate,
        sage_aggregate=sage_aggregate,
        jk_aggregate=jk_aggregate,
        n_out=n_classes,
        sage_project=False,
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    val_loss = train_and_val(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        i_train=splits["train"],
        i_val=splits["val"],
        n_epochs=N_EPOCHS,
        trial=trial,
    )
    return val_loss


def print_callback(study: optuna.Study, frozen_trial: optuna.Trial) -> None:
    print(f"---Trial: {frozen_trial.number}---")
    print(f"Val loss: {frozen_trial.value}")
    if study.best_trial.number == frozen_trial.number:
        print("=> New best.")
    for key, val in frozen_trial.params.items():
        print(f"\t {key}: {val}")


# go!
print("---Run---")
storage = f"sqlite:////{OUTPUT_DIR}/hyperparams.db"
sampler = optuna.samplers.TPESampler(
    n_startup_trials=N_STARTUP_TRIALS,
)
pruner = optuna.pruners.MedianPruner(
    n_warmup_steps=25,
    n_startup_trials=N_STARTUP_TRIALS,
)
study = optuna.create_study(
    study_name="optimize_gnn",
    direction="minimize",
    storage=storage,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
)
study.optimize(
    func=objective,
    n_trials=N_TRIALS,
    callbacks=[print_callback],
    n_jobs=N_JOBS,
)