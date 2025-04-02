import torch


def accuracy(z, targets):
    return torch.mean((torch.sign(z) == targets).float()).item()


def accu(z, targets):
    return torch.mean(((torch.sigmoid(z) > 0.5) == targets).float())


def euclidean_metric(X, v):
    return torch.sum(torch.square(X - v), dim=-1)
