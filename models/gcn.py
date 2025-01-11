# The GCN Model is my solution from practical 4

import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3,
      act_fn: nn.Module = None,
    ):
    super(GCN, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout_ratio
    """
    ## ------ Begin Solution ------ ##
    self.dropout = nn.Dropout(dropout_ratio)
    self.act_fn = act_fn or nn.ReLU()

    if n_layers == 0:
      self.pred = nn.Linear(input_dim, n_classes)
      self.lays = None
    else:
      self.lays = nn.ModuleList()
      self.lays.append(GCNConv(input_dim, hid_dim))
      for _ in range(n_layers):
        self.lays.append(GCNConv(hid_dim, hid_dim))
      self.pred = nn.Linear(hid_dim, n_classes)

    ## ------ End Solution ------ ##

  def forward(self, data) -> torch.Tensor:
    X, A = data.x, data.edge_index
    ## ------ Begin Solution ------ ##
    for layer in self.lays or []:
      X = layer(X, A)
      X = self.act_fn(X)
      X = self.dropout(X)
    X = self.pred(X)
    return X
    ## ------ End Solution ------ ##

  def generate_node_embeddings(self, X, A) -> torch.Tensor:
    return self.forward(X, A)

  def param_init(self):
    ## ------ Begin Solution ------ ##
    nn.init.uniform_(self.pred.weight, -0.1, 0.1)
    if self.pred.bias is not None:
      nn.init.uniform_(self.pred.bias, -0.1, 0.1)

    for layer in self.lays or []:
      nn.init.uniform_(layer.lin.weight, -0.1, 0.1)
      if layer.lin.bias is not None:
        nn.init.uniform_(layer.lin.bias, -0.1, 0.1)
    ## ------ End Solution ------ ##