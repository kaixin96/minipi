from typing import Callable, Optional, Tuple, Type, Union
from collections import defaultdict

import gym3
import numpy as np
import torch as th
from torch.distributed.rpc import RRef

from minipi.algos.common.agent import Actor, Learner, worker_class
from minipi.algos.a2c.policy import A2CBasePolicy
from minipi.buffer.common import Buffer
from minipi.utils.misc import calculate_gae, explained_variance


class A2CActor(Actor):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[A2CBasePolicy],
        policy_kwargs: dict,
        n_steps: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device)
        self.n_steps = n_steps
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda

    def collect(
        self,
        scheduler_step: int,
        buffer: Union[Buffer, RRef],
        learner: Optional[Union[Learner, RRef]] = None,
    ) -> None:
        # Sync parameters if needed
        if learner is not None:
            self.sync_params(learner)
        # Collect a batch of samples
        batch, last_obs, last_first = self.collect_batch()
        # Compute advantage
        batch = self.process_batch(batch, last_obs, last_first)
        # Send data to buffer
        self.add_batch_to_buffer(
            scheduler_step=scheduler_step, batch=batch, size=batch["obs"].shape[1], buffer=buffer,
        )

    def collect_batch(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        Collect a batch of trajectories
        """
        batch = defaultdict(list)
        if self.policy.is_recurrent:
            batch["rnn_states"] = self.policy.rnn_states
        for i in range(self.n_steps):
            reward, obs, first = self.env.observe()
            action, value, logpacs = self.policy.step(obs[None, ...], first[None, ...])
            batch["obs"].append(obs)
            batch["first"].append(first)
            batch["action"].append(action.squeeze(0))
            batch["value"].append(value.squeeze(0))
            batch["logpac"].append(logpacs.squeeze(0))
            self.env.act(action.squeeze(0))
            reward, obs, first = self.env.observe()
            batch["reward"].append(reward)
        if self.policy.is_recurrent:
            if batch["rnn_states"] is None:
                rnn_states = tuple(self.policy.rnn_states)
                batch["rnn_states"] = tuple(th.zeros_like(s) for s in rnn_states)
            batch["rnn_states"] = th.stack(batch["rnn_states"], dim=-1).cpu().numpy()
        # Concatenate
        batch["reward"] = np.asarray(batch["reward"], dtype=float)
        batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
        batch["first"] = np.asarray(batch["first"], dtype=bool)
        batch["action"] = np.asarray(batch["action"])
        batch["value"] = np.asarray(batch["value"], dtype=float)
        batch["logpac"] = np.asarray(batch["logpac"], dtype=float)
        return batch, obs, first

    def process_batch(self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray):
        """
        Process the collected batch, e.g. computing advantages
        """
        last_value = self.policy.value(last_obs[None, ...], last_first[None, ...])
        advs = calculate_gae(
            rewards=batch["reward"],
            values=batch["value"],
            firsts=batch["first"],
            last_value=last_value,
            last_first=last_first,
            discount_gamma=self.discount_gamma,
            gae_lambda=self.gae_lambda,
        )
        batch["adv"] = advs
        return batch


class A2CLearner(Learner):
    def __init__(
        self,
        policy_fn: Type[A2CBasePolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        normalize_adv: bool = False,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(policy_fn, policy_kwargs, device)
        self.optimizer = optimizer_fn(params=self.policy.parameters(), **optimizer_kwargs)
        self.normalize_adv = normalize_adv
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def learn(self, scheduler_step: int, buffer: Union[Buffer, RRef]):
        # Retrieve data from buffer
        if isinstance(buffer, RRef):
            batch = buffer.rpc_sync().get_all()
        else:
            batch = buffer.get_all()
        # Build a dict to save training statistics
        stats_dict = {}
        # Train
        if self.policy.is_recurrent:
            rnn_states = batch["rnn_states"]
        else:
            rnn_states = None
        self.optimizer.zero_grad()
        pg_loss, vf_loss, entropy, extra_out = self.policy.loss(
            obs=batch["obs"],
            advs=batch["adv"],
            firsts=batch["first"],
            actions=batch["action"],
            old_values=batch["value"],
            rnn_states=rnn_states,
            normalize_adv=self.normalize_adv,
        )
        total_loss = pg_loss + self.vf_loss_coef * vf_loss - self.entropy_coef * entropy
        total_loss.backward()
        self.pre_optim_step_hook()
        self.optimizer.step()
        # Saving statistics
        stats_dict["policy_loss"] = pg_loss.item()
        stats_dict["value_loss"] = vf_loss.item()
        stats_dict["entropy"] = entropy.item()
        stats_dict["total_loss"] = total_loss.item()
        stats_dict["explained_variance"] = explained_variance(
            y_pred=batch["value"], y_true=batch["value"] + batch["adv"]
        )
        for key in extra_out:
            stats_dict[key] = extra_out[key].item()
        return stats_dict

    def pre_optim_step_hook(self):
        self.clip_gradient(max_norm=self.max_grad_norm)


class A2CWorker(worker_class(A2CActor, A2CLearner)):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[A2CBasePolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        n_steps: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_adv: bool = False,
        max_grad_norm: Optional[float] = 0.5,
        device: Union[str, th.device] = "cpu",
        worker_weight: float = 1.0,
    ):
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device, worker_weight)
        self.optimizer = optimizer_fn(params=self.policy.parameters(), **optimizer_kwargs)
        self.n_steps = n_steps
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.max_grad_norm = max_grad_norm
