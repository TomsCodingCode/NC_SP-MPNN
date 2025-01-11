import torch
import torch.nn.functional as F

from models.mlp import instantiate_mlp


avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SPN_Layer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_distance,
        eps=0.0,
        trainable_eps=False,
        batch_norm=True,
        device=avail_device,
    ):
        """
        :param in_channels: Dimension size of input. We denote this by I.
        :param out_channels: Dimension size of output: We denote this by O.
        :param max_distance: Maximal shortest distance we're considering.
                             By K we will denote max_distance, so that we don't
                             take the node itself into account (i.e. distance = 0).
        :param eps: The epsilon value used by GIN
        :param trainable_eps: A Boolean specifying whether the epsilon value is trainable
        :param batch_norm: A Boolean specifying whether batch norm is used inside the model MLPs
        """
        super(SPN_Layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_distance = max_distance
        self.device = device

        self.gin_mlp = instantiate_mlp(
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
            final_activation=True,
            batch_norm=batch_norm,
            size=2,
        )

        # Outside aggregation
        self.hop_coef = torch.nn.Parameter(
            torch.randn(self.max_distance).to(device), requires_grad=True
        )

        # epsilon value for self-loop
        self.eps_val = eps
        if trainable_eps:
            self.eps = torch.nn.Parameter(
                torch.randn(1).to(device), requires_grad=True
            )
        else:
            self.eps = eps

    def forward(
            self,
            node_embeddings: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weights: torch.LongTensor):
        """
        :param node_embeddings: A FloatTensor of shape [N, In_dim]
        :param edge_index: A LongTensor of shape [2, #Edges]
        :param edge_weights: The weights by SP_length
        :return: A forward propagation of the input through the HSP layer
        """
        node_cnt = node_embeddings.size(0)  # Number of nodes

        # insight aggregation:
        # sum the embeddings of the i-hop neighbors
        by_hop_aggregates = torch.zeros(
            size=(self.max_distance, node_cnt, self.in_channels), dtype=torch.float
        ).to(self.device)  # A [K, N, I] tensor

        for d in range(1, self.max_distance + 1):
            # Fetch the edges for the current hop distance
            edges = edge_index.T[edge_weights == d].T

            # construct the adjacency matrix for the current hop distance
            values = torch.ones(edges.shape[1], dtype=torch.float, device=self.device)
            sparse_adjacency_d = torch.sparse_coo_tensor(
                indices=edges, values=values, size=(node_cnt, node_cnt), device=self.device
            )  # [N,N] SparseTensor

            # Fetch the embeddings of the neighbors and sum them
            by_hop_aggregates[d - 1, :, :] = torch.sparse.mm(
                sparse_adjacency_d, node_embeddings
            )

        # outside aggregation:
        hop_weights = F.softmax(self.hop_coef, dim=0) # a tensor of shape [K]
        overall_hop_aggregates = (
            (by_hop_aggregates * hop_weights[:, None, None]) # broadcasting the hop weights
            .sum(axis=0) # weighted sum of the embeddings of the neighbors
            .to(self.device)
        ) # this has shape [N, I] now

        out_embeddings = self.gin_mlp(
            (self.eps + 1) * node_embeddings.to(self.device) + overall_hop_aggregates
        )

        return out_embeddings

    def reset_parameters(self):
        for (_, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for x in self.gin_mlp:
            if hasattr(x, "reset_parameters"):
                x.reset_parameters()

        torch.nn.init.normal_(self.hop_coef.data)
        if self.trainable_eps:
            torch.nn.init.normal_(self.eps.data, mean=self.eps_val)

