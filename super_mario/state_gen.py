#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc
import typing
import torch
from functools import reduce
from gym import Env
from gym.wrappers.frame_stack import LazyFrames

from screen_convert import ScreenConverter

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class BaseStateGenerator(abc.ABC):

    def __init__(self, screen_converter: ScreenConverter) -> None:
        super().__init__()
        self._screen_converter = screen_converter

    @abc.abstractmethod
    def gen_state(self, env: Env, frames: LazyFrames) -> typing.Optional[torch.Tensor]:
        pass


class SimpleStateGenerator(BaseStateGenerator):

    def gen_state(self, env: Env, frames: LazyFrames) -> typing.Optional[torch.Tensor]:
        converted_frames = [self._screen_converter.convert(_frame, env) for _frame in frames]
        return reduce(lambda x, y: x - y, converted_frames[::-1])


class StackStateGenerator(BaseStateGenerator):
    def gen_state(self, env: Env, frames: LazyFrames) -> typing.Optional[torch.Tensor]:
        converted_frames = [self._screen_converter.convert(_frame, env) for _frame in frames]

        # frame shape: [C, H, W] , C = frame counts
        return torch.cat(converted_frames)
