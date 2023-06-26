from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from minipi.algos.common.policy import Policy, Extractor
from minipi.network.common import MLP
from minipi.utils.preprocess import dummy_preprocessor


class QNetwork(Extractor):
    def __init__(
        self,
        extractor_fn: Callable[..., nn.Module],
        extractor_kwargs: dict,
        n_actions: int,
        hiddens: Tuple[int, ...] = (),
        dueling: bool = False,
        preprocess_obs_fn: Callable[..., th.Tensor] = dummy_preprocessor,
        preprocess_obs_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            extractor_fn,
            extractor_kwargs,
            preprocess_obs_fn=preprocess_obs_fn,
            preprocess_obs_kwargs=preprocess_obs_kwargs,
        )
        self.qf = MLP(
            input_dim=self.extractor.output_dim,
            hiddens=(*hiddens, n_actions),
            activation=nn.ReLU,
        )
        if dueling:
            self.vf = MLP(
                input_dim=self.extractor.output_dim,
                hiddens=(*hiddens, 1),
                activation=nn.ReLU,
            )
        self.dueling = dueling

    def compute_q_values(self, obs: th.Tensor) -> th.Tensor:
        features, _ = self.extract_features(obs)
        action_scores = self.qf(features)
        if self.dueling:
            state_score = self.vf(features)
            action_scores = action_scores - action_scores.mean(dim=-1, keepdim=True)
            q_values = state_score + action_scores
        else:
            q_values = action_scores
        return q_values


class DQNDiscretePolicy(Policy):
    def __init__(
        self,
        extractor_fn: Callable[..., nn.Module],
        extractor_kwargs: dict,
        n_actions: int,
        hiddens: Tuple[int, ...] = (),
        double_q: bool = False,
        dueling: bool = False,
        preprocess_obs_fn: Callable[..., th.Tensor] = dummy_preprocessor,
        preprocess_obs_kwargs: Optional[dict] = None,
        init_weight_fn: Optional[Callable[..., None]] = None,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(device)
        common_kwargs = {
            "extractor_fn": extractor_fn,
            "extractor_kwargs": extractor_kwargs,
            "n_actions": n_actions,
            "hiddens": hiddens,
            "dueling": dueling,
            "preprocess_obs_fn": preprocess_obs_fn,
            "preprocess_obs_kwargs": preprocess_obs_kwargs,
        }
        self.online_net = QNetwork(**common_kwargs)
        self.target_net = QNetwork(**common_kwargs)

        if init_weight_fn is not None:
            init_weight_fn(self)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.n_actions = n_actions
        self.double_q = double_q

    @th.no_grad()
    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = th.as_tensor(obs).to(self.device).float()
        q_values = self.online_net.compute_q_values(obs[None, ...])
        action = th.argmax(q_values, dim=2)
        return action.cpu().numpy()

    def loss(self, obs, actions, rewards, next_obs, next_firsts, gamma, weights=None):
        # Convert from numpy array to torch tensor
        obs = th.as_tensor(obs).to(self.device).float()
        actions = th.as_tensor(actions).to(self.device)
        rewards = th.as_tensor(rewards).to(self.device).float()
        next_obs = th.as_tensor(next_obs).to(self.device).float()
        next_firsts = th.as_tensor(next_firsts).to(self.device)
        # Q-network evaluation
        q_t = self.online_net.compute_q_values(obs=obs)  # Q(s_t)
        q_t_selected = q_t.gather(dim=2, index=actions[..., None]).squeeze(2)
        # Target Q-network evaluation
        with th.no_grad():
            # Q'(s_{t+1})
            tq_tp1 = self.target_net.compute_q_values(obs=next_obs)
            if self.double_q:
                # Q(s_{t+1})
                q_tp1 = self.online_net.compute_q_values(obs=next_obs)
                actions_selected = q_tp1.max(dim=2).indices
                tq_tp1_selected = tq_tp1.gather(
                    dim=2, index=actions_selected[..., None]
                ).squeeze(2)
            else:
                tq_tp1_selected = tq_tp1.max(dim=2).values
            tq_tp1_masked = (1.0 - next_firsts.float()) * tq_tp1_selected
            td_target = rewards + gamma * tq_tp1_masked
        # Compute loss
        losses = F.smooth_l1_loss(
            input=q_t_selected, target=td_target, reduction="none"
        )
        # Average loss
        if weights is not None:
            weights = th.as_tensor(weights).to(self.device)
            losses = losses * weights
        loss = losses.mean()
        # Calculate additional quantities
        extra_out = {}
        with th.no_grad():
            if weights is not None:
                extra_out["td_error"] = q_t_selected - td_target
        return loss, extra_out

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
