"""
Use GNN embeddings to recommend Amazon products based on cosine similarity of embeddings.
Prints ASIN and Amazon URLs for recommended products.

Note that
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from recommend_gnn.utils import set_safe_globals
from recommend_gnn.model import SageGNN

# params
FILE = Path("/Users/mathis/Code/github/recommend_gnn/results/models/10000_20260319_150300/best_val.pt")
DATA_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/obgn_products_subset10000.pt")
ASIN_FILE = Path("/Users/mathis/Code/github/recommend_gnn/data/nodeidx2asin.csv.gz")
PRODUCT_IDX = 123
OUTPUT = Path("/Users/mathis/Code/github/recommend_gnn/results")
TOP_K = 5
BASE_URL = "www.amazon.com/dp/"

# load model data
contents = torch.load(str(FILE))
state_dict = contents["state_dict"]
hyper_params = contents["hyper_params"]

# load data
set_safe_globals()
data = torch.load(str(DATA_FILE))
if "n_out" not in hyper_params.keys():
    hyper_params["n_out"] = np.unique(data.y.numpy()).size
asin_df = pd.read_csv(ASIN_FILE)


# rebuild model
model = SageGNN(
    n_features=data.x.shape[1],
    n_hidden=hyper_params["n_hidden"],
    depth=hyper_params["depth"],
    n_out=hyper_params["n_out"],
    sage_aggregate=hyper_params["sage_aggregate"],
    jk_aggregate=hyper_params["jk_aggregate"],
    dropout_rate=0,  # doesn't matter for deployment
    sage_project=hyper_params["sage_project"],
    dropout_last=hyper_params["dropout_last"],
)
model.load_state_dict(state_dict)
model.eval()


# get embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(data.x, data.edge_index, depth_from_surface=0)
    embeddings = embeddings.detach().numpy()
print(embeddings.shape)

labels = data.y.detach().numpy()
labels = np.squeeze(labels)

#
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

