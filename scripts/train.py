"""
Train a GNN to predict product category from bag-of-words features and co-purchase information.
"""
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from recommend_gnn.utils import set_safe_globals
from recommend_gnn.model import SageGNN
from recommend_gnn.train import train, test, write_progress

# params
DATA_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/obgn_products_subset10000.pt")
N_HIDDEN = 128
DEPTH = 2
N_EPOCHS = 2000
TEST_FRAC = 0.2
OUTPUT = Path("/Users/mathis/Code/github/recommend_gnn/results")
TB_OUTPUT = Path("/Users/mathis/Code/github/recommend_gnn/results/tb_runs")

# load data
set_safe_globals()
data = torch.load(DATA_FILE)
n_nodes, n_features = data.x.shape
print(f"Nodes (products): {n_nodes:,}")
print(f"Edges: {data.edge_index.shape[1]:,}")
print(f"Features per node: {n_features}")
n_classes = np.unique(data.y).size
print(f"Product classes: {n_classes}")

# format labels
y_true = torch.squeeze(data.y)

# make train mask
i_all_shuffled = np.random.permutation(n_nodes)
n_train = int(n_nodes * (1 - TEST_FRAC))
i_train = i_all_shuffled[:n_train]
i_test = i_all_shuffled[n_train:]

# make model
model = SageGNN(
    n_features=n_features,
    n_hidden=N_HIDDEN,
    depth=DEPTH,
    sage_aggregate="mean",
    jk_aggregate="cat",
    dropout_rate=0.5,
    n_out=n_classes,
)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

log_dir = TB_OUTPUT / f"{datetime.datetime.now()}"
writer = SummaryWriter(str(log_dir))

# train
for i_epoch in range(N_EPOCHS):
    train_loss = train(
        model=model,
        optimizer=optimizer,
        data=data,
        i_train=i_train,
        y_true=y_true,
        loss_fn=loss_fn,
    )
    metrics = test(
        model=model,
        data=data,
        i_train=i_train,
        i_test=i_test,
        loss_fn=loss_fn,
        y_true=y_true,
    )
    write_progress(
        writer=writer,
        i_epoch=i_epoch,
        train_loss=train_loss,
        metrics=metrics,
    )
    print(f"{i_epoch}: {train_loss=:.3f}, test_loss={metrics['test_loss']:.3f}")

