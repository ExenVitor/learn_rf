#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym.core import ObservationWrapper, Env
from gym.wrappers.pixel_observation import PixelObservationWrapper

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class CustomPixelWrapper(ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env, True)
        self._standard_pixel_wrapper = PixelObservationWrapper(env, pixels_only=True)
        self.observation_space = self._standard_pixel_wrapper.observation_space['pixels']

    def observation(self, observation):
        return self._standard_pixel_wrapper.observation(observation=observation)['pixels']



