#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np

from dqn import DQN

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class DDQNAgent(object):
    def __init__(self, channel: int,
                 screen_height: int,
                 screen_width: int,
                 action_num: int,
                 device: torch.device,
                 gamma: float = 0.9,
                 exploration_rate: float = 1.0,
                 exploration_rate_min: float = 0.1,
                 exploration_rate_decay: float = 0.99999975,
                 random_seed: int = 42) -> None:
        super().__init__()
        self._channel = channel
        self._screen_height = screen_height
        self._screen_width = screen_width
        self._action_num = action_num
        self._device = device

        self._gamma = gamma

        self._policy_net = DQN(c=channel, h=screen_height, w=screen_width, outputs=self._action_num).to(self._device)
        self._target_net = DQN(c=channel, h=screen_height, w=screen_width, outputs=self._action_num).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._exploration_rate = exploration_rate
        self._exploration_rate_min = exploration_rate_min
        self._exploration_rate_decay = exploration_rate_decay

        self._random_seed = random_seed
        self._random_state = np.random.RandomState(random_seed)

    def act(self, single_state_tensor: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        if self._random_state.rand() < self._exploration_rate:
            action_tensor = torch.tensor([self._random_state.randint(self._action_num)], dtype=torch.long)
        else:
            with torch.no_grad():
                self._policy_net.eval()
                single_state_tensor = single_state_tensor.to(self._device)
                single_state_tensor = single_state_tensor.unsqueeze(0)
                action_tensor = self._policy_net(single_state_tensor).max(1)[1].cpu()
                self._policy_net.train()

        if not eval_mode:
            new_exploration_rate = self._exploration_rate * self._exploration_rate_decay
            self._exploration_rate = max(self._exploration_rate_min, new_exploration_rate)

        return action_tensor

    def td_estimate(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        # current Q(s_t, a)
        return self._policy_net(state_batch).gather(1, action_batch)

    @torch.no_grad()
    def td_target(self,
                  next_state_batch: torch.Tensor,
                  reward_batch: torch.Tensor,
                  done_batch: torch.Tensor) -> torch.Tensor:
        self._policy_net.eval()
        next_best_action = self._policy_net(next_state_batch).max(1)[1].view(-1, 1).detach()
        self._policy_net.train()

        next_Q = self._target_net(next_state_batch).gather(1, next_best_action)

        return reward_batch + (1 - done_batch.float()) * self._gamma * next_Q

    def sync_target_net(self):
        self._target_net.load_state_dict(self._policy_net.state_dict())

    @property
    def policy_net(self) -> DQN:
        return self._policy_net

    @property
    def config_params(self) -> dict:
        return {
            "channel": self._channel,
            "screen_height": self._screen_height,
            "screen_width": self._screen_width,
            "action_num": self._action_num,
            "gamma": self._gamma,
            "exploration_rate": self._exploration_rate,
            "exploration_rate_min": self._exploration_rate_min,
            "exploration_rate_decay": self._exploration_rate_decay,
            "random_seed": self._random_seed
        }

    def save(self, save_path: str):
        torch.save({
            "policy_net": self._policy_net.state_dict(),
            "target_net": self._policy_net.state_dict(),
            **self.config_params
        }, save_path)

    @classmethod
    def load(cls, save_path: str, device: torch.device, disable_seed: bool = True) -> "DDQNAgent":
        params = torch.load(save_path)
        random_seed = None if disable_seed else params["random_seed"]
        agent = DDQNAgent(channel=params["channel"],
                          screen_height=params["screen_height"],
                          screen_width=params["screen_width"],
                          action_num=params["action_num"],
                          device=device,
                          gamma=params["gamma"],
                          exploration_rate=params["exploration_rate"],
                          exploration_rate_min=params["exploration_rate_min"],
                          exploration_rate_decay=params["exploration_rate_decay"],
                          random_seed=random_seed)

        agent._policy_net.load_state_dict(params["policy_net"])
        agent._target_net.load_state_dict(params["target_net"])
        return agent
