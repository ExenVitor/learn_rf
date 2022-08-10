#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Union, Tuple

from gym import Env
from gym.core import Wrapper, ActType, ObsType, ObservationWrapper

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class SkipFrame(Wrapper):

    def __init__(self, env: Env, skip_frames: int):
        super().__init__(env, False)
        assert skip_frames > 0

        self._skip_frames = skip_frames

    def step(self, action: ActType) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        total_reward = 0.0
        observation = None
        done = False
        info = None
        for i in range(self._skip_frames):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return observation, total_reward, done, info

