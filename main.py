import torch
import torch_geometric
import torch_geometric.datasets as datasets
from torch.nn import ReLU

from models.train_models import get_trained_model

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = datasets.Planetoid(
    root="./",
    name='Cora',
    split="public",
    transform=torch_geometric.transforms.GCNNorm()
)

training_params = {
    "lr": 0.005,  # learning rate
    "weight_decay": 0.0005,  # weight_decay
    "epochs": 100,  # number of total training epochs
    "max_patience": 5, # number of k for early stopping
    "hid_dim": 64, # size of hidden features
    "n_layers": None, # number of layers
}

gcn_params = dict(
    input_dim=dataset.num_features,
    hid_dim=64,
    n_classes=dataset.num_classes,
    n_layers=2,
    dropout_ratio= 0.3,
    #act_fn=ReLU()
)

spn_params = dict(
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    emb_sizes=[64, 64, 64],
    max_distance=5,
    eps=0,
    learnable_eps=False,
    dropout_prob=0.2,
    batch_norm=True,
    device=device
)

#gcn_model = get_trained_model("gcn", dataset.data, gcn_params, training_params)
gcn_params["n_layers"] = 5 # less over-smoothing
#jump_gcn_model = get_trained_model("jump_gcn", dataset.data, gcn_params, training_params)

# to speed up the training, we will precalculate the shortest path distances
def shortest_distances(spn_params, edge_index):
    # k neighborhoods
    k_hops = edge_index
    for k in range(2, spn_params['max_distance'] + 1):
        hop = torch.sparse.mm(k_hops, edge_index)
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
    # return the shortest paths
    return edge_index, edge_weights
e_idx, e_weigh = shortest_distances(spn_params, dataset.data.edge_index)
training_params["predict_func"] = lambda mod, data: mod(data, shortest_paths=(e_idx, e_weigh))
spn_model, acc = get_trained_model("spn", dataset.data, spn_params, training_params)

print("a")