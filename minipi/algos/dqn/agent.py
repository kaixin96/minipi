from typing import Callable, Optional, Type, Union

import gym3
import numpy as np
import torch as th
from torch.distributed.rpc import RRef

from minipi.algos.common.agent import Actor, Learner, worker_class
from minipi.algos.dqn.policy import DQNDiscretePolicy
from minipi.buffer.common import ReplayBuffer
from minipi.utils.schedulers import Schedulable, get_scheduler


class DQNActor(Actor):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[DQNDiscretePolicy],
        policy_kwargs: dict,
        n_step_return: int,
        exploration_eps: Schedulable,
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device)
        self.n_step_return = n_step_return
        self.exploration_eps = get_scheduler(exploration_eps)

    def collect(
        self,
        scheduler_step: int,
        buffer: Union[ReplayBuffer, RRef],
        learner_rref: Optional[RRef] = None,
    ):
        # Sync parameters if needed
        if learner_rref is not None:
            self.sync_params(learner_rref)
        # Collect samples
        batch = self.collect_batch(scheduler_step)
        # Send data to buffer
        self.add_batch_to_buffer(
            scheduler_step=scheduler_step,
            batch=batch,
            size=batch["obs"].shape[1],
            buffer=buffer,
        )

    def collect_batch(self, scheduler_step: int) -> dict:
        batch = {}
        _, obs, _ = self.env.observe()
        eps = self.exploration_eps.value(step=scheduler_step)
        determ_action = self.policy.step(obs=obs)
        random_action = np.random.randint(0, self.policy.n_actions, determ_action.shape)
        choose_random = np.random.rand(*random_action.shape) < eps
        action = np.where(choose_random, random_action, determ_action)
        self.env.act(action.squeeze(0))
        reward, next_obs, next_first = self.env.observe()
        # TODO: unnormalize obs / rew if needed (before add into buffer)
        batch["obs"] = obs[None, ...]
        batch["actions"] = action
        batch["rewards"] = reward[None, ...]
        batch["next_obs"] = next_obs[None, ...]
        batch["next_firsts"] = next_first[None, ...]
        return batch


class DQNLearner(Learner):
    def __init__(
        self,
        policy_fn: Type[DQNDiscretePolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        batch_size: int,
        discount_gamma: float = 0.99,
        max_grad_norm: Optional[float] = 40.0,
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__(policy_fn, policy_kwargs, device)
        self.optimizer = optimizer_fn(
            params=self.policy.online_net.parameters(), **optimizer_kwargs
        )
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.max_grad_norm = max_grad_norm

    def learn(self, scheduler_step: int, buffer: Union[ReplayBuffer, RRef]) -> dict:
        # Retrieve data from buffer
        if isinstance(buffer, ReplayBuffer):
            batch = buffer.sample(scheduler_step, self.batch_size)
        else:
            batch = buffer.rpc_sync().sample(scheduler_step, self.batch_size)
        # Build a dict to save training statistics
        stats_dict = {}
        # Train
        self.optimizer.zero_grad()
        loss, extra_out = self.policy.loss(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_obs=batch["next_obs"],
            next_firsts=batch["next_firsts"],
            gamma=self.discount_gamma,
            weights=batch.get("weights", None),
        )
        loss.backward()
        self.pre_optim_step_hook()
        self.optimizer.step()
        # Update priorities if needed
        if "weights" in batch:
            indices = batch["indices"]
            priorities = extra_out["td_error"].abs().cpu().numpy().squeeze(0)
            extra_out["td_error"] = extra_out["td_error"].mean()
            if isinstance(buffer, ReplayBuffer):
                buffer.update_priorities(
                    scheduler_step=scheduler_step,
                    indices=indices,
                    priorities=priorities,
                )
            else:
                buffer.rpc_sync().update_priorities(
                    scheduler_step=scheduler_step,
                    indices=indices,
                    priorities=priorities,
                )
        # Saving statistics
        stats_dict["loss"] = loss.item()
        for key in extra_out:
            stats_dict[key] = extra_out[key].item()
        return stats_dict


class DQNWorker(worker_class(DQNActor, DQNLearner)):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[DQNDiscretePolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        n_step_return: int,
        exploration_eps: Schedulable,
        batch_size: int,
        discount_gamma: float = 0.99,
        max_grad_norm: Optional[float] = 40.0,
        device: Union[th.device, str] = "cpu",
        worker_weight: float = 1.0,
    ):

        super().__init__(
            env_fn, env_kwargs, policy_fn, policy_kwargs, device, worker_weight,
        )
        self.optimizer = optimizer_fn(
            params=self.policy.online_net.parameters(), **optimizer_kwargs
        )
        self.n_step_return = n_step_return
        self.exploration_eps = get_scheduler(exploration_eps)
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.max_grad_norm = max_grad_norm
