"""
Use GNN embeddings to recommend Amazon products based on cosine similarity of embeddings.
Prints ASIN and Amazon URLs for recommended products.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from recommend_gnn.utils import set_safe_globals
from recommend_gnn.model import SageGNN


# params
PRODUCT_IDX = 123
TOP_K = 5
BASE_URL = "www.amazon.com/dp/"


def main() -> None:
    # set paths
    current_dir = Path.cwd()
    model_file = current_dir / "results" / "model.pt"
    data_file = current_dir / "data" / "obgn_products_subset10000.pt"
    asin_file = current_dir / "data" / "nodeidx2asin.csv.gz"
    output_dir = current_dir / "results"

    # load model data
    contents = torch.load(str(model_file))
    state_dict = contents["state_dict"]
    hyper_params = contents["hyper_params"]

    # load data
    set_safe_globals()
    data = torch.load(str(data_file))
    if "n_out" not in hyper_params.keys():
        hyper_params["n_out"] = np.unique(data.y.numpy()).size
    asin_df = pd.read_csv(asin_file)


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

    # log source product
    save_file = output_dir / "product_recommendations.txt"
    print(f"Saving product recommendations to {save_file}")
    asin = asin_df.loc[asin_df["node idx"] == PRODUCT_IDX, "asin"].values[0]
    product_url = f"https://{BASE_URL}/{asin}"
    txt = f"Finding recommendations for product {PRODUCT_IDX} ({product_url})"
    print(txt)
    with open(save_file, mode="w") as file:
        file.write(f"{txt}\n")

    # recommend products
    scores = cosine_similarity(embeddings)
    scores = scores[PRODUCT_IDX, :]
    sorted_idx = np.argsort(scores)
    top_idx = sorted_idx[-TOP_K:]
    for i in top_idx:
        if i != PRODUCT_IDX:
            asin = asin_df.loc[asin_df["node idx"] == i, "asin"].values[0]
            product_url = f"https://{BASE_URL}/{asin}"
            this_score = scores[i]
            txt = f"Recommending product {i} with similarity {this_score:.3f}: {product_url}"
            print(txt)
            with open(save_file, mode="a") as file:
                file.write(f"{txt}\n")


if __name__ == "__main__":
    main()
