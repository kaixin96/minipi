from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from minipi.algos.common.actor_critic import Actor, QCritic
from minipi.algos.common.policy import Policy
from minipi.utils.preprocess import dummy_preprocessor
from minipi.utils.misc import polyak_update


class TD3ContinuousPolicy(Policy):
    def __init__(
        self,
        extractor_fn: Callable[..., nn.Module],
        extractor_kwargs: dict,
        action_dim: int,
        n_critics: int = 2,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        preprocess_obs_fn: Callable[..., th.Tensor] = dummy_preprocessor,
        preprocess_obs_kwargs: Optional[dict] = None,
        target_update_ratio: float = 1.0,
        target_policy_noise_std: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        init_weight_fn: Optional[Callable[..., None]] = None,
        device: Union[str, th.device] = "cpu",
    ) -> None:
        super().__init__(device)

        # Build actor
        common_kwargs = {
            "extractor_fn": extractor_fn,
            "extractor_kwargs": extractor_kwargs,
            "preprocess_obs_fn": preprocess_obs_fn,
            "preprocess_obs_kwargs": preprocess_obs_kwargs,
        }
        actor_kwargs = common_kwargs.copy()
        actor_kwargs.update(
            {"squash_output": True, "hiddens": actor_hiddens, "n_outputs": action_dim}
        )
        self.actor = Actor(**actor_kwargs)
        self.actor_target = Actor(**actor_kwargs)

        # Build critics
        critic_kwargs = common_kwargs.copy()
        critic_kwargs.update({"hiddens": critic_hiddens, "action_dim": action_dim})
        self.critic = nn.ModuleList(
            [QCritic(**critic_kwargs) for _ in range(n_critics)]
        )
        self.critic_target = nn.ModuleList(
            [QCritic(**critic_kwargs) for _ in range(n_critics)]
        )

        # Initialize weights if needed
        if init_weight_fn is not None:
            init_weight_fn(self)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.action_dim = action_dim
        self.target_update_ratio = target_update_ratio
        self.target_policy_noise_std = target_policy_noise_std
        self.target_policy_noise_clip = target_policy_noise_clip

    @th.no_grad()
    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = th.as_tensor(obs).to(self.device).float()
        actions, _ = self.actor(obs)
        return actions.cpu().numpy()

    def critic_loss(self, obs, actions, next_obs, rewards, next_firsts, gamma, weights):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        actions = th.as_tensor(actions).to(self.device).float()
        next_obs = th.as_tensor(next_obs).to(self.device).float()
        rewards = th.as_tensor(rewards).to(self.device).unsqueeze(-1).float()
        next_firsts = th.as_tensor(next_firsts).to(self.device).unsqueeze(-1).float()

        # Compute target Q-values
        with th.no_grad():
            # Add clipped noise to the target policy for smoothing the value estimate
            noise = actions.clone().data.normal_(0, self.target_policy_noise_std)
            noise.clamp_(-self.target_policy_noise_clip, self.target_policy_noise_clip)
            next_actions, _ = self.actor_target(next_obs)
            next_actions = th.clamp(next_actions + noise, -1, 1)
            next_q_values = [qf(next_obs, next_actions)[0] for qf in self.critic_target]
            next_q_values = th.stack(next_q_values, dim=-1).min(dim=-1).values
            target_q_values = rewards + (1.0 - next_firsts) * gamma * next_q_values

        # Compute current Q-values
        current_q_values = [qf(obs, actions)[0] for qf in self.critic]

        # Compute critic loss
        critic_loss = sum([F.mse_loss(qv, target_q_values) for qv in current_q_values])

        extra_out = {}
        return critic_loss, extra_out

    def actor_loss(self, obs):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        # Compute actor loss only using first critic to reduce computation
        actions, _ = self.actor(obs)
        q_values, _ = self.critic[0](obs, actions)
        actor_loss = -q_values.mean()
        extra_out = {}
        return actor_loss, extra_out

    def update_target_net(self):
        polyak_update(
            params=self.actor.parameters(),
            target_params=self.actor_target.parameters(),
            tau=self.target_update_ratio,
        )
        polyak_update(
            params=self.critic.parameters(),
            target_params=self.critic_target.parameters(),
            tau=self.target_update_ratio,
        )
