import torch
from torch_geometric.data.data import GlobalStorage, DataEdgeAttr, DataTensorAttr

def set_safe_globals() -> None:
    """For some reason, we need to set some torch globals to safe to load eg OGB data"""
    torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
