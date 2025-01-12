# The SPN model is based on the NetHSP_GIN model implemented in the SP-MPNN paper:
# https://github.com/radoslav11/SP-MPNN/blob/main/src/models/hsp_gin.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from .mlp import instantiate_mlp
from .spn_layer import SPN_Layer

avail_device = "cuda" if torch.cuda.is_available() else "cpu"

class SPN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes=None,
        max_distance=5,
        eps=0,
        learnable_eps=False,
        dropout_prob=0.2,
        device=avail_device,
        batch_norm=True,
    ):
        """
        Constructs a SPN model.
        :param num_features: Number of input features
        :param num_classes: Number of output classes
        :param emb_sizes: List of embedding sizes for each layer
        :param max_distance: Maximal shortest distance we're considering.
                                By K we will denote max_distance, so that we don't
                                take the node itself into account (i.e. distance = 0).
        :param eps: The epsilon value used by GIN
        :param dropout_prob: Dropout probability
        """
        super(SPN, self).__init__()
        if emb_sizes is None:
            emb_sizes = [64, 64, 64]

        self.max_distance = max_distance
        self.num_layers = len(emb_sizes) - 1
        self.eps = eps
        self.device = device

        self.initial_mlp = instantiate_mlp(
            in_channels=num_features,
            out_channels=emb_sizes[0],
            device=device,
            batch_norm=batch_norm,
            final_activation=True,
            size=1,
        )

        sp_mp_layers = [] # shortest path message passing layers
        for i in range(self.num_layers):
            hsp_layer = SPN_Layer(
                in_channels=emb_sizes[i],
                out_channels=emb_sizes[i + 1],
                max_distance=self.max_distance,
                eps=self.eps,
                trainable_eps=learnable_eps,
                batch_norm=batch_norm,
                device=device
            ).to(device)
            sp_mp_layers.append(hsp_layer)

        self.sp_mp_layers = ModuleList(sp_mp_layers)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        self.prediction_head = instantiate_mlp(
            in_channels=emb_sizes[-1],
            out_channels=num_classes,
            device=device,
            final_activation=False,
            batch_norm=batch_norm,
            size=2,
        )

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.initial_mlp.reset_parameters()

        for layer in self.sp_mp_layers:
            layer.reset_parameters()

        self.prediction_head.reset_parameters()


    def forward(self, data):
        x_feat: torch.Tensor = data.x.to(self.device)
        edge_index: torch.Tensor = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1), device=self.device), device=self.device)

        # Shortest path calculation
        # k neighborhoods
        k_hops = edge_index
        for k in range(2, self.max_distance + 1):
            hop = torch.sparse.mm(k_hops, edge_index)
            hop = (hop - k_hops * torch.inf).coalesce()
            adj = hop.indices().T[hop.values() > 0].T
            k_hops += torch.sparse_coo_tensor(
                adj,
                torch.ones(adj.size(1), device=self.device) * k,
                hop.shape,
                device=self.device
            )

        # remove the diagonal and copy to appropriate variables
        k_hops = k_hops.coalesce()
        mask = k_hops.indices()[0] != k_hops.indices()[1]
        edge_index = k_hops.indices()[:, mask].to(self.device)
        edge_weights = k_hops.values()[mask].to(self.device)

        # Input encoding
        x_feat = self.initial_mlp(x_feat)

        for layer in self.sp_mp_layers:
            x_feat = layer(
                node_embeddings=x_feat,
                edge_index=edge_index,
                edge_weights=edge_weights,
            ).to(self.device)
            x_feat = self.dropout(x_feat)

        return self.prediction_head(x_feat)

    def log_hop_weights(self, neptune_client, exp_dir):
        if self.outside_aggr in ["weight"]:
            for i, layer in enumerate(self.sp_mp_layers):
                data = layer.hop_coef.data
                soft_data = F.softmax(data, dim=0)
                for d, (v, sv) in enumerate(zip(data, soft_data), 1):
                    log_dir = exp_dir + "/conv_" + str(i) + "/" + "weight_" + str(d)
                    neptune_client[log_dir].log(v)
                    soft_log_dir = (
                        exp_dir + "/conv_" + str(i) + "/" + "soft_weight_" + str(d)
                    )
                    neptune_client[soft_log_dir].log(sv)
