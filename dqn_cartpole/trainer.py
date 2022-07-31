#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
from itertools import count
import typing

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_memory import ReplayMemory, Transition
from state_gen import BaseStateGenerator

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

DEFAULT_BATCH_SIZE = 128
DEFAULT_GAMMA = 0.999
DEFAULT_EPS_START = 0.9
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY_STEPS = 200
DEFAULT_TARGET_UPDATE_EPISODES = 10


class DQNTrainer(object):
    def __init__(self, env: gym.Env,
                 state_generator: BaseStateGenerator,
                 n_actions: int,
                 device: torch.device,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 gamma: int = DEFAULT_GAMMA,
                 eps_start: int = DEFAULT_EPS_START,
                 eps_end: int = DEFAULT_EPS_END,
                 eps_decay_steps: int = DEFAULT_EPS_DECAY_STEPS,
                 target_update_episodes: int = DEFAULT_TARGET_UPDATE_EPISODES,
                 random_seed: int = 42) -> None:
        super().__init__()

        self._env = env
        self._n_actions = n_actions
        self._device = device
        self._batch_size = batch_size
        self._gamma = gamma
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay_steps = eps_decay_steps
        self._target_update_episodes = target_update_episodes

        self._state_generator = state_generator
        observation = env.reset()
        init_state = self._state_generator.gen_state(env=self._env, frames=observation, terminated=False)
        _, channel, screen_height, screen_width = init_state.shape

        self._policy_net = DQN(c=channel, h=screen_height, w=screen_width, outputs=n_actions).to(device=self._device)
        self._target_net = DQN(c=channel, h=screen_height, w=screen_width, outputs=n_actions).to(device=self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.RMSprop(self._policy_net.parameters())
        self._memory = ReplayMemory(10000)

        self._steps_done = 0
        self._episode_durations = []
        random.seed(random_seed)

    # def _gen_state(self, frames: LazyFrames, terminated: bool) -> typing.Optional[torch.Tensor]:
    #     if terminated:
    #         return None
    #     converted_frames = [self._screen_converter.convert(_frame, self._env) for _frame in frames]
    #     return reduce(lambda x, y: x - y, converted_frames[::-1])

    def _select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = (self._eps_end + (self._eps_start - self._eps_end) *
                         math.exp(-1. * self._steps_done / self._eps_decay_steps))
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state_tensor = state_tensor.to(self._device)
                return self._policy_net(state_tensor).max(1)[1].view(1, 1).cpu()
        else:
            return torch.tensor([[self._env.action_space.sample()]], dtype=torch.long)

    def _optimize_model(self):
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self._device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self._device)

        state_batch = torch.cat(batch.state).to(self._device)
        action_batch = torch.cat(batch.action).to(self._device)
        reward_batch = torch.cat(batch.reward).to(self._device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self._policy_net.parameters(), clip_value=1)
        # for param in self._policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def train(self, num_episodes: int, episode_end_callback: typing.Optional[typing.Callable] = None):
        self._steps_done = 0
        self._episode_durations = []

        for i in range(num_episodes):
            observation = self._env.reset()
            state_tensor = self._state_generator.gen_state(env=self._env, frames=observation, terminated=False)

            for t in count():
                action_tensor = self._select_action(state_tensor=state_tensor)
                observation, reward, terminated, _, _ = self._env.step(action_tensor.item())
                reward_tensor = torch.tensor([reward])

                next_state_tensor = self._state_generator.gen_state(env=self._env, frames=observation,
                                                                    terminated=terminated)

                self._memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

                state_tensor = next_state_tensor

                self._optimize_model()
                if terminated:
                    self._episode_durations.append(t + 1)
                    if episode_end_callback is not None:
                        episode_end_callback(i, t + 1, self._episode_durations)
                    break

            if i % self._target_update_episodes == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
