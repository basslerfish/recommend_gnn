Use a Graph Neural Network (GNN) to recommend products from the obgn-products dataset of Amazon products.

**Illustration**

<img 
    src="images/gnn_product_rec.png"
    alt="Explanation of product recommendation"
    width="600"
    style="display:block; margin-top:10px; margin-bottom:10px;"
/>

**How it works**
- Train a GNN (`torch`,`torch_geometric`) to predict product class from product features and co-purchases (́`scripts/train_model.py`)
- Optionally use hyperparameter tuning (`optuna`) to find best model parameters (`scripts/optimize_hyperparams.py`, `scripts/train_best.py`)
- Run `scripts/main.py` to get product recommendations given a product IDX

**How to run**
- Clone repository
- Easiest is to use docker:
  - `docker build -t recommend_gnn .`
  - `docker run recommend_gnn`

The dataset this repo relies on (obgn-product) was collected up until 2014.
Therefore, not all product URLs may link to valid products.
Inspired by the book "Graph neural networks in Action" by Broadwater and Stillman (https://github.com/keitabroadwater/gnns_in_action/).
