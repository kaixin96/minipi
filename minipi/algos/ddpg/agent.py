from typing import Callable, Optional, Type, Union

import gym3
import numpy as np
import torch as th
from torch.distributed.rpc import RRef

from minipi.algos.common.agent import Actor, Learner, worker_class
from minipi.utils.noise import ActionNoise
from minipi.algos.ddpg.policy import DDPGContinuousPolicy
from minipi.buffer.common import ReplayBuffer


class DDPGActor(Actor):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[DDPGContinuousPolicy],
        policy_kwargs: dict,
        action_noise_fn: Optional[Type[ActionNoise]] = None,
        action_noise_kwargs: Optional[dict] = None,
        warmup_steps: int = 0,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device)
        self.warmup_steps = warmup_steps
        self.action_noise = None
        if action_noise_fn is not None:
            self.action_noise = action_noise_fn(**(action_noise_kwargs or {}))

    def collect(
        self,
        scheduler_step: int,
        buffer: Union[ReplayBuffer, RRef],
        learner: Optional[Union[Learner, RRef]] = None,
    ) -> None:
        # Sync parameters if needed
        if learner is not None:
            self.sync_params(learner)
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
        reward, obs, first = self.env.observe()
        if scheduler_step < self.warmup_steps:
            action = np.random.uniform(
                -1.0, 1.0, size=(self.env.num, self.policy.action_dim)
            )
        else:
            action = self.policy.step(obs=obs[None, ...])
            action = action.squeeze(0)
            if self.action_noise is not None:
                self.action_noise.reset(flags=first, size=action.shape)
                noise = self.action_noise(size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
        self.env.act(action)
        reward, next_obs, next_first = self.env.observe()
        # Prepend timestep dimension T
        batch["obs"] = obs[None, ...]
        batch["actions"] = action[None, ...]
        batch["rewards"] = reward[None, ...]
        batch["next_obs"] = next_obs[None, ...]
        batch["next_firsts"] = next_first[None, ...]
        return batch


class DDPGLearner(Learner):
    def __init__(
        self,
        policy_fn: Type[DDPGContinuousPolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        batch_size: int,
        discount_gamma: float = 0.99,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(policy_fn, policy_kwargs, device)
        self.actor_optimizer = optimizer_fn(
            params=self.policy.actor.parameters(), **optimizer_kwargs["actor"]
        )
        self.critic_optimizer = optimizer_fn(
            params=self.policy.critic.parameters(), **optimizer_kwargs["critic"]
        )
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma

    def learn(self, scheduler_step: int, buffer: Union[ReplayBuffer, RRef]) -> dict:
        # Retrieve data from buffer
        if isinstance(buffer, RRef):
            batch = buffer.rpc_sync().sample(scheduler_step, self.batch_size)
        else:
            batch = buffer.sample(scheduler_step, self.batch_size)

        # Build a dict to save training statistics
        stats_dict = {}

        # Train critic
        critic_loss, critic_extra_out = self.policy.critic_loss(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_obs=batch["next_obs"],
            next_firsts=batch["next_firsts"],
            gamma=self.discount_gamma,
            weights=batch.get("weights", None),
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train actor
        actor_loss, actor_extra_out = self.policy.actor_loss(obs=batch["obs"])
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.policy.update_target_net()

        # Update priorities if needed
        if "weights" in batch and "td_error" in critic_extra_out:
            indices = batch["indices"]
            priorities = critic_extra_out["td_error"].abs().cpu().numpy()
            critic_extra_out["td_error"] = critic_extra_out["td_error"].mean()
            if isinstance(buffer, ReplayBuffer):
                buffer.update_priorities(
                    current_timestep=scheduler_step,
                    indices=indices,
                    priorities=priorities,
                )
            else:
                buffer.rpc_sync().update_priorities(
                    current_timestep=scheduler_step,
                    indices=indices,
                    priorities=priorities,
                )

        # Saving statistics
        stats_dict["critic_loss"] = critic_loss.item()
        stats_dict["actor_loss"] = actor_loss.item()
        for key, value in critic_extra_out.items():
            stats_dict[key] = value.item()
        for key, value in actor_extra_out.items():
            stats_dict[key] = value.item()

        return stats_dict


class DDPGWorker(worker_class(DDPGActor, DDPGLearner)):
    def __init__(
        self,
        env_fn: Callable[..., gym3.Env],
        env_kwargs: dict,
        policy_fn: Type[DDPGContinuousPolicy],
        policy_kwargs: dict,
        optimizer_fn: Type[th.optim.Optimizer],
        optimizer_kwargs: dict,
        batch_size: int,
        action_noise_fn: Optional[Type[ActionNoise]] = None,
        action_noise_kwargs: Optional[dict] = None,
        warmup_steps: int = 0,
        discount_gamma: float = 0.99,
        device: Union[str, th.device] = "cpu",
        worker_weight: float = 1.0,
    ):
        super().__init__(
            env_fn, env_kwargs, policy_fn, policy_kwargs, device, worker_weight
        )
        self.actor_optimizer = optimizer_fn(
            params=self.policy.actor.parameters(), **optimizer_kwargs["actor"]
        )
        self.critic_optimizer = optimizer_fn(
            params=self.policy.critic.parameters(), **optimizer_kwargs["critic"]
        )
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma
        self.warmup_steps = warmup_steps
        self.action_noise = None
        if action_noise_fn is not None:
            self.action_noise = action_noise_fn(**(action_noise_kwargs or {}))
