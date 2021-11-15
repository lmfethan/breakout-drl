from typing import (
    Tuple,
)

import torch
import numpy as np

from .utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            alpha: float,
            beta: float,
            beta_increment: float,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.p = torch.zeros((capacity, 1), requires_grad=False)
        self.p.data[0][0] = 1
        self.index = range(capacity)

    def __len__(self) -> int:
        return self.__size

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:

        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        maximum = self.p.data.max()
        self.p.data[self.__pos, 0] = maximum

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int):
        # indices = torch.randint(0, high=self.__size, size=(batch_size,))
        self.beta = min(1, self.beta + self.beta_increment)

        p_alpha = torch.pow(self.p.squeeze(1), self.alpha)
        p_alpha_sum = p_alpha.sum()
        prob = p_alpha / p_alpha_sum
        indices = torch.multinomial(prob, batch_size).detach()

        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        p_batch = self.p[indices]
        b_weights = torch.pow(self.__size * p_batch, -self.beta).unsqueeze(1)
        b_weights = b_weights / b_weights.max()
        return indices, b_state, b_action, b_reward, b_next, b_done, b_weights.to(self.__device)

    def update(self, indices, error):
        self.p[indices].data = torch.abs(error)


