from typing import Dict

import numpy as np

from minipi.buffer.common import ReplayBuffer
from minipi.buffer.segment_tree import SumSegmentTree
from minipi.utils.schedulers import Schedulable, get_scheduler

class PrioritizedReplayBuffer(ReplayBuffer):
    # TO be updated with new T x B x shape format
    def __init__(
        self,
        capacity: int,
        sequence_length: int = 1,
        alpha: Schedulable = 0.6,
        beta: Schedulable = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__(capacity=capacity, sequence_length=sequence_length)
        self.alpha = get_scheduler(alpha)
        self.beta = get_scheduler(beta)
        self.eps = eps
        self.p_alpha = SumSegmentTree(size=self.capacity)
        self.max_priority = 1.0

    def add(self, scheduler_step: int, data: Dict[str, np.ndarray], idx: int, size: int):
        super().add(scheduler_step=scheduler_step, data=data, idx=idx, size=size)
        idx_end = idx + size
        if idx_end <= self.capacity:
            indices = np.r_[idx:idx_end]
        else:
            indices = np.r_[idx : self.capacity, 0 : idx_end % self.capacity]
        current_alpha = self.alpha.value(step=scheduler_step)
        self.p_alpha[indices] = self.max_priority ** current_alpha

    def sample_indices(self, batch_size: int):
        """
        Range [0, ptotal] is divided equally into k ranges,
        then a value is uniformly sampled from each range.
        """
        p_alpha_total = self.p_alpha.sum()
        p_alpha_range = np.linspace(0, p_alpha_total, num=batch_size, endpoint=False)
        shift = np.random.random_sample(size=batch_size) * p_alpha_total / batch_size
        mass = p_alpha_range + shift
        indices = self.p_alpha.find_prefixsum_idx(prefixsum=mass)
        # Clip to handle the case where mass[i] is very close to p_alpha_total
        # In that case, indices[i] will be self.p_alpha.capacity
        indices = np.clip(indices, None, self.real_size - 1)
        return indices

    def sample(self, scheduler_step: int, batch_size: int):
        indices = self.sample_indices(batch_size=batch_size)
        probs = self.p_alpha[indices] / self.p_alpha.sum()
        current_beta = self.beta.value(step=scheduler_step)
        weights = (probs * self.real_size) ** (-current_beta)
        weights /= np.max(weights)
        batch = self.get_by_indices(indices=indices)
        batch["weights"] = weights
        batch["indices"] = indices
        return batch

    def update_priorities(self, scheduler_step: int, indices, priorities):
        priorities += self.eps

        assert len(indices) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(indices) >= 0
        assert np.max(indices) < self.real_size

        current_alpha = self.alpha.value(step=scheduler_step)
        self.p_alpha[indices] = priorities ** current_alpha
        self.max_priority = max(self.max_priority, np.max(priorities))


