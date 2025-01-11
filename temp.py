import torch

def make_line(length: int) -> torch.Tensor:
    """
    Creates the adjacency matrix of a line graph.
    :param length: The number of nodes in the line graph.
    """
    adj = torch.zeros((length, length))
    for i in range(length - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj

l_4 = make_line(4)
l_4 = l_4.to_sparse()
 
# k = 1
x1 = l_4
#print(x1.to_dense())

# k = 2
x2 = x1 @ l_4
x2 = (x2 - x1 * torch.inf).coalesce()
adj = x2.indices().T[x2.values() > 0].T
x2 = torch.sparse_coo_tensor(adj, torch.ones(adj.size(1)), x2.shape)
#print(x2.to_dense())

# k = 3
x3 = x2 @ l_4
x3 = (x3 - (x2 + x1) * torch.inf).coalesce()
adj = x3.indices().T[x3.values() > 0].T
x3 = torch.sparse_coo_tensor(adj, torch.ones(adj.size(1)), x3.shape)
#print(x3.to_dense())

# max_sitance = 3
# adj = make_line(5)
# k_hops = [adj]
# for k in range(2, max_sitance):
#     k_hop = k_hops[-1] @ adj
#     k_hop = (k_hop - sum(k_hops) * torch.inf).coalesce()
#     k_adj = k_hop.indices().T[k_hop.values() > 0].T
#     k_hop = torch.sparse_coo_tensor(k_adj, torch.ones(k_adj.size(1)) * k, k_hop.shape)


max_sitance = 3
adj = make_line(5).to_sparse()
k_hops = adj
for k in range(2, max_sitance):
    hop = k_hops @ adj
    hop = (hop - k_hops * torch.inf).coalesce()
    k_adj = hop.indices().T[hop.values() > 0].T
    k_hops += torch.sparse_coo_tensor(k_adj, torch.ones(k_adj.size(1)) * k, hop.shape)

# remove the diagonal
k_hops = k_hops.coalesce()
diag_mask = k_hops.indices()[0] == k_hops.indices()[1]
k_hops = torch.sparse_coo_tensor(k_hops.indices()[:, ~diag_mask], k_hops.values()[~diag_mask], k_hops.shape)

print(k_hops.to_dense())