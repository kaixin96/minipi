import torch
import torch.nn as nn


def xavier_uniform_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def lasagne_orthogonal_init(m, scale=1.0):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        lasagne_orthogonal(m.weight, scale)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def lasagne_orthogonal(tensor, scale=1.0):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)
    u, _, v = torch.svd(flattened)
    q = u if u.shape == flattened.shape else v  # pick the one with the correct shape
    q = q.reshape(tensor.shape)
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(scale)
    return tensor

