#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import datetime as dt
from dataclasses import dataclass
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import typing
from itertools import count
from torch.utils.tensorboard import SummaryWriter

from state_gen import BaseStateGenerator
from replay_memory import ReplayMemory, Transition
from agent import DDQNAgent
from utils import check_and_mkdirs

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

DEFAULT_BATCH_SIZE = 32
DEFAULT_MEMORY_SIZE = int(1e5)
DEFAULT_SAVE_EVERY_STEPS = 5e5
DEFAULT_BURNIN_STEPS = 1e4  # min. experiences before training
DEFAULT_LEARN_EVERY_STEPS = 3
DEFAULT_SYNC_EVERY_STEPS = 1e4

DEFAULT_LR = 0.00025


@dataclass
class LearnResult(object):
    avg_td_est: float
    loss: float


class TrainRecorder(object):
    DEFAULT_MA_SIZE = 100

    def __init__(self, ma_size: int = DEFAULT_MA_SIZE) -> None:
        super().__init__()
        self._ma_size = ma_size

        self._ep_rewards = []
        self._ep_lengths = []
        self._ep_avg_losses = []
        self._ep_avg_qs = []

        # moving averages
        self._ma_ep_rewards = []
        self._ma_ep_lengths = []
        self._ma_avg_losses = []
        self._ma_avg_qs = []

        self._cur_ep_rewards = []
        self._cur_ep_losses = []
        self._cur_ep_qs = []

    def init_episode_state(self):
        self._cur_ep_rewards = []
        self._cur_ep_losses = []
        self._cur_ep_qs = []

    def log_step(self, reward: float, loss: typing.Optional[float], q: typing.Optional[float]):
        self._cur_ep_rewards.append(reward)
        if loss is not None:
            self._cur_ep_losses.append(loss)
        if q is not None:
            self._cur_ep_qs.append(q)

    def _update_ma(self, ma_list: list, src_list: list):
        if len(src_list) >= self._ma_size:
            ma_list.append(np.mean(src_list[-self._ma_size:]))

    def log_episode(self, ep_duration: int):
        self._ep_lengths.append(ep_duration)
        self._ep_rewards.append(sum(self._cur_ep_rewards))
        self._ep_avg_losses.append(np.mean(self._cur_ep_losses))
        self._ep_avg_qs.append(np.mean(self._cur_ep_qs))

        self._update_ma(self._ma_ep_lengths, self._ep_lengths)
        self._update_ma(self._ma_ep_rewards, self._ep_rewards)
        self._update_ma(self._ma_avg_losses, self._ep_avg_losses)
        self._update_ma(self._ma_avg_qs, self._ep_avg_qs)

        self.init_episode_state()

    @classmethod
    def _gen_value_pair(cls, name: str, ma_list: list, src_list: list) -> dict:
        if len(src_list) == 0:
            return {}
        value = src_list[-1]
        ma_value = ma_list[-1] if len(ma_list) > 0 else None

        pair_dict = {
            name: value
        }
        if ma_value is not None:
            pair_dict[f"{name}-MA"] = ma_value

        return {
            name: pair_dict
        }

    def get_snapshot(self) -> dict:

        result = {
            **self._gen_value_pair("EP length", self._ma_ep_lengths, self._ep_lengths),
            **self._gen_value_pair("EP reward", self._ma_ep_rewards, self._ep_rewards),
            **self._gen_value_pair("EP avg loss", self._ma_avg_losses, self._ep_avg_losses),
            **self._gen_value_pair("EP avg q", self._ma_avg_qs, self._ep_avg_qs)
        }

        return result


class CheckpointManager(object):
    CHECKPOINT_FILE_FORMAT = "agent_{}.chkpt"
    INFO_FILE_FORMAT = "checkpoint_info.json"

    def __init__(self, base_output_dir: str, model_id: str) -> None:
        super().__init__()
        self._base_output_dir = base_output_dir
        self._model_id = model_id
        self._model_path = os.path.join(self._base_output_dir, self._model_id)
        self._info_file_path = os.path.join(self._model_path, self.INFO_FILE_FORMAT)

    def load_info(self) -> dict:

        try:
            with open(self._info_file_path, "r") as fp:
                info_dict = json.load(fp)
        except Exception as e:
            info_dict = {
                "best_checkpoint_name": None,
                "train_steps": 0
            }
        return info_dict

    def save_checkpoint(self, agent: DDQNAgent, checkpoint_idx: int, steps: int, is_best: bool):
        checkpoint_file_name = self.CHECKPOINT_FILE_FORMAT.format(checkpoint_idx)
        agent.save(os.path.join(self._model_path, checkpoint_file_name))

        info_dict = self.load_info()
        info_dict["train_steps"] = steps
        if is_best or info_dict["best_checkpoint_name"] is None:
            info_dict["best_checkpoint_name"] = checkpoint_file_name

        with open(self._info_file_path, 'w') as fp:
            json.dump(info_dict, fp, indent=4)

    def get_best_checkpoint_path(self) -> str:
        info_dict = self.load_info()
        checkpoint_name = info_dict["best_checkpoint_name"]
        if checkpoint_name is None:
            raise RuntimeError("Checkpoint not exist")
        return os.path.join(self._model_path, checkpoint_name)


class DQNTrainer(object):
    def __init__(self,
                 env: gym.Env,
                 state_generator: BaseStateGenerator,
                 device: torch.device,
                 base_output_dir: str,
                 model_tag: str,
                 lr: float = DEFAULT_LR,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 memory_size: int = DEFAULT_MEMORY_SIZE,
                 save_every_steps: int = DEFAULT_SAVE_EVERY_STEPS,
                 burnin_steps: int = DEFAULT_BURNIN_STEPS,
                 learn_every_steps: int = DEFAULT_LEARN_EVERY_STEPS,
                 sync_every_steps: int = DEFAULT_SYNC_EVERY_STEPS,

                 ) -> None:
        super().__init__()
        self._env = env
        self._state_generator = state_generator
        self._device = device
        self._base_output_dir = base_output_dir
        model_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_tag}"
        self._output_dir = os.path.join(self._base_output_dir,
                                        model_id)
        check_and_mkdirs(self._output_dir)
        self._writer = SummaryWriter(log_dir=self._output_dir)

        self._lr = lr
        self._batch_size = batch_size
        self._memory_size = memory_size
        self._save_every_steps = save_every_steps
        self._burnin_steps = burnin_steps
        self._learn_every_steps = learn_every_steps
        self._sync_every_steps = sync_every_steps

        self._steps = 0

        observation = env.reset()
        init_state = self._state_generator.gen_state(env=self._env, frames=observation)
        channel, screen_height, screen_width = init_state.shape
        self._agent = DDQNAgent(channel=channel, screen_height=screen_height, screen_width=screen_width,
                                action_num=env.action_space.n, device=self._device)

        self._optimizer = optim.Adam(self._agent.policy_net.parameters(), lr=lr)
        self._loss_fn = nn.SmoothL1Loss()
        self._memory = ReplayMemory(self._memory_size)

        self._writer.add_graph(self._agent.policy_net,
                               init_state.to(self._device).unsqueeze(0).expand(self._batch_size, *init_state.shape))
        self._writer.flush()

        self._train_recoder = TrainRecorder()
        self._checkpoint_manager = CheckpointManager(base_output_dir=self._base_output_dir, model_id=model_id)

    @property
    def agent(self) -> DDQNAgent:
        return self._agent

    @property
    def model_output_dir(self) -> str:
        return self._output_dir

    @property
    def config_params(self) -> dict:
        return {
            "lr": self._lr,
            "batch_size": self._batch_size,
            "memory_size": self._memory_size,
            "save_every_steps": self._save_every_steps,
            "burnin_steps": self._burnin_steps,
            "learn_every_steps": self._learn_every_steps,
            "sync_every_steps": self._sync_every_steps,
            "state_generator_cls": self._state_generator.__class__.__name__
        }

    def gen_checkpoint_idx(self) -> int:
        return self._steps // self._save_every_steps

    def _save_checkpoint(self):
        # Consider early stopping
        self._checkpoint_manager.save_checkpoint(agent=self._agent, checkpoint_idx=self.gen_checkpoint_idx(),
                                                 steps=self._steps, is_best=True)

    def _save_params(self):
        params_dict = {
            "agent_params": self._agent.config_params,
            "trainer_params": self.config_params
        }
        with open(os.path.join(self._output_dir, "params.json"), 'w') as fp:
            json.dump(params_dict, fp, indent=4)

    def _learn(self) -> typing.Optional[LearnResult]:
        if self._steps < self._burnin_steps:
            return None

        if self._steps % self._save_every_steps == 0:
            self._save_checkpoint()

        if self._steps % self._sync_every_steps == 0:
            self._agent.sync_target_net()

        if self._steps % self._learn_every_steps != 0:
            return None

        transitions = self._memory.sample(self._batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self._device)
        action_batch = torch.stack(batch.action).to(self._device)
        reward_batch = torch.stack(batch.reward).to(self._device)
        next_state_batch = torch.stack(batch.next_state).to(self._device)
        done_batch = torch.stack(batch.done).to(self._device)

        td_est = self._agent.td_estimate(state_batch=state_batch, action_batch=action_batch)

        td_target = self._agent.td_target(next_state_batch=next_state_batch, reward_batch=reward_batch,
                                          done_batch=done_batch)

        loss = self._loss_fn(td_est, td_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return LearnResult(avg_td_est=td_est.mean().item(), loss=loss.item())

    def train(self, num_episodes: int, episode_end_callback: typing.Optional[typing.Callable] = None):
        self._steps = 0
        self._save_params()
        for i in range(num_episodes):
            observation = self._env.reset()
            state_tensor = self._state_generator.gen_state(env=self._env, frames=observation)

            for t in count():
                action_tensor = self._agent.act(state_tensor, eval_mode=False)
                self._steps += 1

                observation, reward, done, info = self._env.step(action_tensor.item())

                next_state_tensor = self._state_generator.gen_state(env=self._env, frames=observation)
                reward_tensor = torch.tensor([reward])
                done_tensor = torch.tensor([done])

                self._memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)

                learn_result = self._learn()
                _loss = None
                _q = None
                if learn_result is not None:
                    _loss = learn_result.loss
                    _q = learn_result.avg_td_est

                self._train_recoder.log_step(reward=reward, loss=_loss, q=_q)

                state_tensor = next_state_tensor

                if done or info['flag_get']:
                    if episode_end_callback is not None:
                        episode_end_callback(i, t + 1, self._steps)
                    self._train_recoder.log_episode(t + 1)
                    ep_stat = self._train_recoder.get_snapshot()
                    for name, scaler_dict in ep_stat.items():
                        self._writer.add_scalars(name, scaler_dict, i)
                    self._writer.flush()
                    break

        self._save_checkpoint()
