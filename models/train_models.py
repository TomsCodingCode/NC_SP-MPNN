import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm

from .gcn import GCN
from .jump_gcn import JumpKnowGCN
from .spn import SPN


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


def get_trained_model(model: str, data, model_params: dict, params: dict):
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

    optimizer = Adam(model.parameters(), lr=params.get("lr", 0.001), weight_decay=params.get("weight_decay", 0))

    # training iteration
    max_acc = 0
    validation_prog = []
    progress_iterator = tqdm(range(params.get("epochs", 100)), desc='Loss: unknown')
    for epoch in progress_iterator:
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        progress_iterator.set_description(f'Loss: {loss.item():.4f}')

        val_acc = evaluate(model, data, data.val_mask)
        validation_prog.append(val_acc)
        if val_acc > max_acc:
            max_acc = val_acc
            best_params = model.state_dict()
            patience = 0
        else:
            patience += 1
            # with max_patience = -1 there is no early stopping
            if patience == params.get("max_patience", -1):
                break
        
        if params.get("verbose", False):
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Val Acc: {val_acc}")

    if best_params is not None:
        model.load_state_dict(best_params)
    train_acc = evaluate(model, data, data.train_mask)
    test_acc = evaluate(model, data, data.test_mask)
    print(f'Training ended after {epoch + 1} Epochs:')
    print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {max_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    return model, validation_prog
