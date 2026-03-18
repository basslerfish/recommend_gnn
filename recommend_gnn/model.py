"""
A GNN to analyze the product co-purchase graph.
"""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, JumpingKnowledge

class SageGNN(nn.Module):
    """GraphSAGE-based GNN with flexible depth, aggregate functions and dropout rate."""
    def __init__(
            self,
            n_features: int,
            n_hidden: int,
            n_out: int,
            depth: int,
            sage_aggregate: str,
            sage_project: bool,
            jk_aggregate: str,
            dropout_rate: float,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        # make conv layers
        layers = []
        for i_layer in range(depth):
            if i_layer == 0:
                conv = SAGEConv(n_features, n_hidden, aggr=sage_aggregate, project=sage_project)
            else:
                conv = SAGEConv(n_hidden, n_hidden, aggr=sage_aggregate, project=sage_project)
            layers.append(conv)
        layers = nn.ModuleList(layers)
        self.conv_layers = layers

        if jk_aggregate == "lstm":
            self.jk = JumpingKnowledge(mode=jk_aggregate, channels=n_hidden, num_layers=1)
        else:
            self.jk = JumpingKnowledge(mode=jk_aggregate)
        if jk_aggregate == "cat":
            n_jk = int(len(layers) * n_hidden)
        else:
            n_jk = n_hidden
        self.fc1 = nn.Linear(n_jk, n_out)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            return_embed: bool = False,
    ) -> torch.Tensor:
        """Compute either embeddings or class predictions (not softmaxed)"""
        intermediates = []
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout_rate, train=self.training)
            intermediates.append(x)
        x = self.jk(intermediates)
        if return_embed:
            return x
        else:
            x = self.fc1(x)
            return x




