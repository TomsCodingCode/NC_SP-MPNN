import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.gcn import GCN
from models.jump_gcn import JumpKnowGCN
from models.spn import SPN


def evaluate(
    model,
    data,
    mask
):
    model.eval()
    with torch.no_grad():
      logits = model(data)
      preds = logits.argmax(dim=1)
      acc = (preds[mask] == data.y[mask]).sum().item() / mask.sum().item()
    return acc


def get_trained_model(model: str, data, model_params, params):
    """
    Returns a trained model.
    :param model: The model to be trained
    :param model_params: The parameters of the model
    :param params: The training parameters
    :return: A trained model
    """
    if model == "gcn":
        model = GCN(**model_params)
    elif model == "spn":
        model = SPN(**model_params)
    elif model == "jump_gcn":
        model = JumpKnowGCN(**model_params)
    else:
        raise ValueError(f"Model {model} not supported.")
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    data = data.to(device)

    optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # training iteration
    max_acc = 0
    for epoch in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask)
        if val_acc > max_acc:
            max_acc = val_acc
            best_params = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience == params["max_patience"]:
                pass
                #break
        
        if type(model) == SPN:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Val Acc: {val_acc}")

    if best_params is not None:
        model.load_state_dict(best_params)
    print(f'Training ended after {epoch} Epochs with accuracy {max_acc}')
    return model
