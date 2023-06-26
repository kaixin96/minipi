import torch as th


def dummy_preprocessor(tensor: th.Tensor) -> th.Tensor:
    return tensor


def scale_preprocessor(tensor: th.Tensor, max_value: float) -> th.Tensor:
    return tensor / max_value