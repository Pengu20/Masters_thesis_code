from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.learning import ReplayMemory


class BaseLearning(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_action(o: np.ndarray) -> torch.FloatTensor:
        raise NotImplementedError(
            "Method 'select_action' must be implemented in learning algorithm."
        )

    @abstractmethod
    def optimize(
        batch_size: int, tensorboard_writer: SummaryWriter = None
    ) -> torch.FloatTensor:
        raise NotImplementedError(
            "Method 'optimize' must be implemented in learning algorithm."
        )

    @abstractmethod
    def a_noisy(batch_size: int) -> torch.FloatTensor:
        raise NotImplementedError(
            "Method 'a_noisy' must be implemented in learning algorithm."
        )

    @abstractmethod
    def save(name: str) -> None:
        raise NotImplementedError(
            "Method 'save' must be implemented in learning algorithm."
        )

    @abstractmethod
    def load(name: str) -> torch.FloatTensor:
        raise NotImplementedError(
            "Method 'load' must be implemented in learning algorithm."
        )

    @property
    @abstractmethod
    def name() -> str:
        raise NotImplementedError(
            "Property 'name' must be implemented in learning algorithm."
        )

    @property
    @abstractmethod
    def replay_buffer() -> ReplayMemory:
        raise NotImplementedError(
            "Property 'replay_buffer' must be implemented in learning algorithm."
        )

    @replay_buffer.setter
    @abstractmethod
    def replay_buffer(new_replay_buffer: ReplayMemory) -> None:
        raise NotImplementedError(
            "Property 'replay_buffer' must be implemented in learning algorithm."
        )

    @property
    @abstractmethod
    def a_max() -> int:
        raise NotImplementedError(
            "Property 'a_max' must be implemented in learning algorithm."
        )

    @property
    @abstractmethod
    def a_dim() -> int:
        raise NotImplementedError(
            "Property 'a_dim' must be implemented in learning algorithm."
        )
