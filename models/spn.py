# The SPN model is based on the NetHSP_GIN model implemented in the SP-MPNN paper:
# https://github.com/radoslav11/SP-MPNN/blob/main/src/models/hsp_gin.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleList

from .mlp import instantiate_mlp
from .spn_layer import SPN_Layer
from ..utils.shortest_paths import shortest_distances

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
        initial_mlp_size=1,
        prediction_head_size=2,
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
            size=initial_mlp_size,
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
            size=prediction_head_size,
        )

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.initial_mlp.reset_parameters()

        for layer in self.sp_mp_layers:
            layer.reset_parameters()

        self.prediction_head.reset_parameters()


    def forward(self, data, shortest_paths = None) -> torch.Tensor:
        x_feat = self.generate_node_embeddings(data, shortest_paths)
        return self.prediction_head(x_feat)
    
    def generate_node_embeddings(self, data, shortest_paths = None) -> torch.Tensor:
        X = data.x.to(self.device)
        A = data.edge_index.to(self.device)

        # Shortest path calculation
        if shortest_paths is None:
            edge_index, edge_weights = shortest_distances(self.max_distance, A)
        else:
            edge_index, edge_weights = shortest_paths
        
        # Input encoding
        X = self.initial_mlp(X)

        # Message passing
        for layer in self.sp_mp_layers:
            X = layer(
                node_embeddings=X,
                edge_index=edge_index,
                edge_weights=edge_weights,
            ).to(self.device)
            X = self.dropout(X)
        
        return X

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
