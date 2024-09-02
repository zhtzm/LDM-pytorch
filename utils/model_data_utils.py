import torch


def disable_train_mode(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
