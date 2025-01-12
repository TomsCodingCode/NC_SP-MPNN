from typing import Tuple

import torch


# k neighborhoods
def shortest_distances(max_distance: int, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the shortest distances between nodes in the graph upto a maximal distance of K.
    The calculated distances are returned as an edge_index tensor containing the multi hop edges and a weight tensor denoting the lengths.
    :param max_distance: Maximal shortest distance we're considering.
    :param edge_index: Edge index tensor of the graph. A LongTensor of shape [2, #Edges]
    :return: A tuple containing the edge index tensor and the edge weights tensor. Shapes [2, #Edges] and [#Edges]
    """
    adjacency = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), device=edge_index.device)
    k_hops = adjacency.clone()
    for k in range(2, max_distance + 1):
        hop = torch.sparse.mm(k_hops, adjacency)
        hop = (hop - k_hops * torch.inf).coalesce()
        adj = hop.indices().T[hop.values() > 0].T
        k_hops += torch.sparse_coo_tensor(
            adj,
            torch.ones(adj.size(1), device=edge_index.device) * k,
            hop.shape,
            device=edge_index.device
        )

    # remove the diagonal and copy to appropriate variables
    k_hops = k_hops.coalesce()
    mask = k_hops.indices()[0] != k_hops.indices()[1]
    edge_index = k_hops.indices()[:, mask].to(edge_index.device)
    edge_weights = k_hops.values()[mask].to(edge_index.device)

    return edge_index, edge_weights