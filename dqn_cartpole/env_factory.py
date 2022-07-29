#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from custom_pixel_wrapper import CustomPixelWrapper
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.human_rendering import HumanRendering

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class EnvFactory(object):

    def __init__(self, env_id: str, random_seed: int = 42) -> None:
        super().__init__()
        self._env_id = env_id
        self._random_seed = random_seed

    def gen_env(self, num_stack: int = 2, human_render_mode: bool = False) -> gym.Env:
        base_env = gym.make(self._env_id, new_step_api=True, render_mode="single_rgb_array")
        base_env.action_space.seed(self._random_seed)
        base_env.reset(seed=self._random_seed)

        wrapper = FrameStack(CustomPixelWrapper(base_env), num_stack=num_stack, new_step_api=True)
        if human_render_mode:
            wrapper = HumanRendering(wrapper)

        return wrapper
