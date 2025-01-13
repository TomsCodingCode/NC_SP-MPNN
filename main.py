import torch
import torch_geometric
import torch_geometric.datasets as datasets
from torch.nn import ReLU

from models.train_models import get_trained_model
from utils.shortest_paths import shortest_distance_prediction_function

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
    "epochs": 100,  # maximal number of total training epochs
    "max_patience": 10, # patience for early stopping
}

gcn_params = dict(
    input_dim=dataset.num_features,
    hid_dim=64,
    n_classes=dataset.num_classes,
    n_layers=5,
    dropout_ratio= 0.3,
)

spn_params = dict(
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    emb_sizes=[64, 64, 64],
    max_distance=5,
    eps=0,
    learnable_eps=True,
    dropout_prob=0.2,
    batch_norm=True,
    device=device
)

gcn_model = get_trained_model("gcn", dataset.data, gcn_params, training_params)
jump_gcn_model = get_trained_model("jump_gcn", dataset.data, gcn_params, training_params)

# to speed up the training, we will precalculate the shortest path distances
training_params["predict_func"] = shortest_distance_prediction_function(spn_params["max_distance"], dataset.data.edge_index)
spn_model, acc = get_trained_model("spn", dataset.data, spn_params, training_params)
