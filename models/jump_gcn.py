import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class JumpKnowGCN(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3):
    super(JumpKnowGCN, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout ratio
    """
    ## ------ Begin Solution ------ ##
    self.dropout_ratio = dropout_ratio

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
    X = self.generate_node_embeddings(data.x, data.edge_index)
    X = self.pred(X)
    return X
    ## ------ End Solution ------ ##

  def generate_node_embeddings(self, data) -> torch.Tensor:
    X, A = data.x, data.edge_index
    results = [] # <-- new
    for layer in self.lays or []:
      X = layer(X, A)
      X = F.relu(X)
      X = F.dropout(X, p=self.dropout_ratio, training=self.training)
      results.append(X)
    # v new v
    if results:
      X = torch.stack(results).max(dim=0).values
    return X

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