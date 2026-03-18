"""
Use GNN embeddings to recommend Amazon products based on cosine similarity of embeddings.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from recommend_gnn.utils import set_safe_globals
from recommend_gnn.model import SageGNN

# params
FILE = Path("/Users/mathis/Code/github/recommend_gnn/results/models/20260318_164308/best_val.pt")
DATA_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/obgn_products_subset10000.pt")
ASIN_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/nodeidx2asin.csv.gz")
N_FEATURES = 100
N_CLASSES = 32
PRODUCT_IDX = 123
OUTPUT = Path("/Users/mathis/Code/github/recommend_gnn/results")
TOP_K = 5
BASE_URL = "www.amazon.com/dp/"

# load model data
contents = torch.load(str(FILE))
state_dict = contents["state_dict"]
hyper_params = contents["hyper_params"]

# rebuild model
model = SageGNN(
    n_features=N_FEATURES,
    n_hidden=hyper_params["n_hidden"],
    depth=hyper_params["depth"],
    n_out=N_CLASSES,
    sage_aggregate=hyper_params["sage_aggregate"],
    jk_aggregate=hyper_params["jk_aggregate"],
    dropout_rate=0,
    sage_project=hyper_params["sage_project"],
)
model.load_state_dict(state_dict)
model.eval()

# load data
set_safe_globals()
data = torch.load(str(DATA_FILE))

# get embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(data.x, data.edge_index, depth_from_surface=0)
    embeddings = embeddings.detach().numpy()
print(embeddings.shape)

labels = data.y.detach().numpy()
labels = np.squeeze(labels)

reduced = PCA().fit_transform(embeddings)

fig, ax = plt.subplots()
for l in np.unique(labels):
    is_l = labels == l
    ax.scatter(reduced[is_l, 0], reduced[is_l, 1])
plt.savefig(OUTPUT / "pca.png")
plt.close(fig)

#
asin_df = pd.read_csv(ASIN_FILE)
print(asin_df)

print(f"Finding recommendations for product {PRODUCT_IDX}")
scores = cosine_similarity(embeddings)
scores = scores[PRODUCT_IDX, :]
sorted_idx = np.argsort(scores)
top_idx = sorted_idx[-TOP_K:]
for i in top_idx:
    if i != PRODUCT_IDX:
        asin = asin_df.loc[asin_df["node idx"] == i, "asin"].values[0]
        product_url = f"https://{BASE_URL}/{asin}"
        print(f"Recommending product {i}: {product_url}")

