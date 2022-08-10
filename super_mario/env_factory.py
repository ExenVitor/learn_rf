#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers.frame_stack import FrameStack
from wrappers import SkipFrame

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class EnvFactory(object):

    def __init__(self, env_id: str, movements: typing.List[typing.List[str]],
                 skip_frames: int = 4, num_stack: int = 4) -> None:
        super().__init__()
        self._env_id = env_id
        self._movements = movements
        assert self._movements is not None
        assert len(self._movements) > 0
        for sub_l in movements:
            assert sub_l in COMPLEX_MOVEMENT

        self._skip_frames = skip_frames
        self._num_stack = num_stack
        assert self._num_stack > 0

    def gen_env(self, random_seed: typing.Optional[int] = 42) -> gym.Env:
        env = gym_super_mario_bros.make(self._env_id)
        env.seed(random_seed)
        env = JoypadSpace(env, self._movements)
        if self._skip_frames > 0:
            env = SkipFrame(env, skip_frames=self._skip_frames)
        env = FrameStack(env, num_stack=self._num_stack)
        env.action_space.seed(random_seed)

        return env

    @property
    def config_params(self) -> dict:
        return {
            "env_id": self._env_id,
            "movements": self._movements,
            "skip_frames": self._skip_frames,
            "num_stacks": self._num_stack,
        }
