from typing import Dict, Iterator, Union

import numpy as np
import torch as th

from minipi.utils.schedulers import ConstantScheduler


@th.no_grad()
def infer_fc_input_dim(module, input_shape):
    dummy_input = th.rand(2, *input_shape)
    dummy_output = module(dummy_input)
    infer_dim = th.flatten(dummy_output, start_dim=1).shape[-1]
    return infer_dim


def clamp(input, min, max):
    # Currently torch.clamp() does not support tensor arguments for min/max
    # while torch.min() / max() does not support float arguments
    if isinstance(min, th.Tensor) and isinstance(max, th.Tensor):
        clipped = th.max(th.min(input, max), min)
    else:
        clipped = th.clamp(input, min=min, max=max)
    return clipped


def calculate_gae(rewards, values, firsts, last_value, last_first, discount_gamma, gae_lambda):
    # borrow implementation from OpenAI's baselines
    n_steps = len(rewards)
    advs = np.zeros_like(rewards)
    lastadv = 0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - last_first
            nextvalues = last_value
        else:
            nextnonterminal = 1.0 - firsts[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + discount_gamma * nextvalues * nextnonterminal - values[t]
        advs[t] = delta + discount_gamma * gae_lambda * nextnonterminal * lastadv
        lastadv = advs[t]
    return advs


def get_scheduler(value: Union[Dict, float]):
    if isinstance(value, float):
        scheduler = ConstantScheduler(value=value)
    elif isinstance(value, Dict):
        assert "scheduler_fn" in value and "scheduler_kwargs" in value
        scheduler = value["scheduler_fn"](**value["scheduler_kwargs"])
    else:
        raise TypeError("value should be either Dict or float")
    return scheduler


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=float)
        self.var = np.ones(shape, dtype=float)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0, dtype=float)
        batch_var = np.var(x, axis=0, dtype=float)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count


@th.no_grad()
def polyak_update(
    params: Iterator[th.Tensor], target_params: Iterator[th.Tensor], tau: float
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params`` in-place.
    TODO: ensure same length in zip
    """
    for param, target_param in zip(params, target_params):
        target_param.data.mul_(1 - tau)
        th.add(target_param.data, param.data, alpha=tau, out=target_param.data)
