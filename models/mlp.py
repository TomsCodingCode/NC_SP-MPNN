import torch
from torch.nn import ModuleList, Linear, BatchNorm1d, ReLU, Sequential


avail_device = "cuda" if torch.cuda.is_available() else "cpu"


def instantiate_mlp(
    in_channels,
    out_channels,
    device=avail_device,
    final_activation=True,
    batch_norm=True,
    size=2,
):
    """
    Instantiates a multi-layer perceptron with ReLU activations and optional batch normalization.
    The hidden layer sizes are the same as the output size.
    :param in_channels: Dimension of input features
    :param out_channels: Dimension of output features
    :param device: Device to which the model is sent
    :param final_activation: Whether to include the final activation
    :param batch_norm: Whether to include batch normalization
    :param size: Number of layers in the MLP
    :return: A multi-layer perceptron
    """
    modules = ModuleList()
    in_dim, out_dim = in_channels, out_channels
    
    for _ in range(size):
        modules.append(Linear(in_dim, out_dim).to(device))
        if batch_norm:
            modules.append(BatchNorm1d(out_dim).to(device))
        modules.append(ReLU().to(device))
        in_dim = out_dim
    
    if not final_activation:
        modules.pop(-1)
    return Sequential(*modules).to(device)
