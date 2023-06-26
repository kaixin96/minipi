import abc
from typing import Dict, Sequence, KeysView
from threading import Lock

import numpy as np


class Buffer:
    def __init__(self, capacity: int, sequence_length: int = 1) -> None:
        """A FIFO memory buffer that stores sequences of transitions.

        The transitions are organized in the shape of :math:`(T, N, *)`,
        where :math:`T` is the length of each sequence
        and :math:`N` is the number of sequences.

        Args:
            capacity: The maximum number of sequences the buffer.
            sequence_length: The length of each sequence in the buffer. Defaults to 1.

        """
        self._lock = Lock()
        self.capacity = capacity
        self.sequence_length = sequence_length
        assert self.capacity > 0
        self.real_size = 0
        self.next_idx = 0
        self.storage = {}

    def __len__(self) -> int:
        """
        Returns:
            The number of sequences in the buffer.

        """
        return self.real_size

    def keys(self) -> KeysView:
        """
        Returns:
            The keys of the stored data.

        """
        return self.storage.keys()

    def update_next_idx(self, size):
        """Update the index before adding data, so that different actors do not overwrite.

        Args:
            size: The number of sequences in the data.
        
        Returns:
            The original index before updating.

        """
        with self._lock:
            idx = self.next_idx
            self.next_idx = (self.next_idx + size) % self.capacity
            return idx

    def add(self, scheduler_step: int, data: Dict[str, np.ndarray], idx: int, size: int):
        """Add data (sequences of transitions) to the buffer.

        Args:
            scheduler_step: The step input for the schedulers.
            data: The transition data to be added.
            idx: The starting index in the buffer where the data will be added
            size: The number of sequences in the data.

        """
        # Initialize if buffer is empty
        if not self.storage:
            for key, value in data.items():
                self.storage[key] = np.zeros(
                    shape=(value.shape[0], self.capacity, *value.shape[2:]), dtype=value.dtype,
                )
        # Add data
        idx_end = idx + size
        for key, value in data.items():
            if idx_end <= self.capacity:
                self.storage[key][:, idx:idx_end] = value
            else:
                first, second = np.split(value, (self.capacity - idx,), axis=1)
                self.storage[key][:, idx:] = first
                self.storage[key][:, : idx_end - self.capacity] = second
        # Set size
        with self._lock:
            self.real_size = min(self.real_size + size, self.capacity)

    def get_by_indices(self, indices: Sequence[int]) -> Dict[str, np.ndarray]:
        """Retrieve a batch of data from the buffer by indices

        Args:
            indices: The indices of the sequences.

        Returns:
            The batch of data retrieved from the buffer.

        """
        indices = np.asarray(indices, dtype=int)
        return {key: value[:, indices] for key, value in self.storage.items()}

    def get_all(self):
        """
        Returns:
            The entire buffer.
        """
        return {key: value[:, : self.real_size] for key, value in self.storage.items()}


class ReplayBuffer(Buffer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample(self, scheduler_step: int, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of data from the buffer

        Args:
            scheduler_step: The step input for the schedulers.
            batch_size: The number of sequences to be sampled.

        Returns:
            The sampled batch of data.
        """
        raise NotImplementedError


class UniformReplayBuffer(ReplayBuffer):
    def sample(self, scheduler_step: int, batch_size: int):
        """Uniformly sample a batch of data from the buffer

        Args:
            scheduler_step: The step input for the schedulers.
            batch_size: The number of sequences to be sampled.

        Returns:
            The sampled batch of data.
        """
        indices = np.random.randint(self.real_size, size=batch_size)
        batch = self.get_by_indices(indices)
        return batch

