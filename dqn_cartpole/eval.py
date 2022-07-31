#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
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

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

DEFAULT_OUTPUT_DIR = "outputs"


@dataclass
class EvalResult(object):
    episode_durations: typing.List[int]

    def __post_init__(self):
        self._df = pd.DataFrame(data={'durations': self.episode_durations})

    @property
    def stat_dict(self) -> dict:
        stat_se = self._df['durations'].describe()
        return stat_se.to_dict()

    @property
    def hist_img(self):
        return self._df.hist()

    def save(self, name: str, output_base_dir: str = DEFAULT_OUTPUT_DIR):
        output_dir = path.join(output_base_dir, name)
        check_and_mkdirs(output_dir)

        with open(path.join(output_dir, "stat.json"), "w") as fp:
            json.dump(self.stat_dict, fp, indent=4)

        plt.figure()
        hist_img = self.hist_img
        plt.savefig(path.join(output_dir, "hist.jpg"))


class Evaluator(object):
    def __init__(self,
                 env: gym.Env,
                 state_generator: BaseStateGenerator,
                 device: torch.device,
                 policy_net: nn.Module) -> None:
        super().__init__()
        self._env = env
        self._state_generator = state_generator
        self._device = device
        self._policy_net = policy_net

    def run(self, eval_rounds: int = 100) -> EvalResult:
        durations = []
        for i_round in range(eval_rounds):
            observation = self._env.reset()
            state = self._state_generator.gen_state(env=self._env, frames=observation, terminated=False)
            for i_step in count():
                with torch.no_grad():
                    action = self._policy_net(state.to(self._device)).max(1)[1].item()
                observation, reward, terminated, _, _ = self._env.step(action)
                if terminated:
                    durations.append(i_step + 1)
                    break
                state = self._state_generator.gen_state(env=self._env, frames=observation, terminated=terminated)

        return EvalResult(durations)
