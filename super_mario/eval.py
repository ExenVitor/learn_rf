#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import gym
from gym.wrappers import RecordVideo
import torch
import torch.nn as nn
import typing
from dataclasses import dataclass
import pandas as pd
from os import path
import json
import matplotlib.pyplot as plt
from itertools import count

from state_gen import BaseStateGenerator
from utils import check_and_mkdirs
from agent import DDQNAgent

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

DEFAULT_OUTPUT_DIR = "outputs"


@dataclass
class EvalResult(object):
    episode_durations: typing.List[int]
    episode_rewards: typing.List[float]
    episode_flag_get: typing.List[bool]

    def __post_init__(self):
        self._df = pd.DataFrame(data={'durations': self.episode_durations,
                                      'rewards': self.episode_rewards,
                                      'flag_get': self.episode_flag_get})
        self._df.index.name = "episode_idx"

    @property
    def stat_dict(self) -> dict:
        stat_df = self._df.describe()
        return stat_df.to_dict()

    def hist_img(self, col_name: str):
        return self._df[col_name].hist()

    def save(self, model_output_dir: str):
        self._df.to_csv(path.join(model_output_dir, "eval_records.csv"))

        with open(path.join(model_output_dir, "stat.json"), "w") as fp:
            json.dump(self.stat_dict, fp, indent=4)

        for col in self._df.columns:
            try:
                plt.figure()
                hist_img = self.hist_img(col)
                plt.savefig(path.join(model_output_dir, f"{col}_hist.jpg"))
            except Exception as e:
                pass


class Evaluator(object):
    def __init__(self,
                 env: gym.Env,
                 state_generator: BaseStateGenerator,
                 device: torch.device,
                 agent: DDQNAgent,
                 video_record_dir: typing.Optional[str] = None) -> None:
        super().__init__()
        self._env = env
        self._state_generator = state_generator
        self._device = device
        self._agent = agent
        self._video_record_dir = video_record_dir

    def run(self, eval_rounds: int = 100, render: bool = False) -> EvalResult:
        durations = []
        rewards = []
        flag_get = []
        env = self._env
        if self._video_record_dir:
            env = RecordVideo(env, video_folder=self._video_record_dir, episode_trigger=lambda eps_i: True)
        for i_round in range(eval_rounds):
            observation = env.reset()
            state = self._state_generator.gen_state(env=env, frames=observation)
            cur_reward = 0
            for i_step in count():
                action = self._agent.act(state.to(self._device), eval_mode=True).item()
                observation, reward, done, info = env.step(action)
                cur_reward += reward
                if render:
                    env.render()
                if done:
                    durations.append(i_step + 1)
                    rewards.append(cur_reward)
                    flag_get.append(info["flag_get"])
                    break
                state = self._state_generator.gen_state(env=env, frames=observation)

        eval_result = EvalResult(durations, rewards, flag_get)
        if self._video_record_dir:
            eval_result.save(self._video_record_dir)

        return eval_result
