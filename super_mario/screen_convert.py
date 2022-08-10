#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing
import numpy as np
import torch
import torchvision.transforms as T
import abc
from gym import Env

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class ScreenConverter(abc.ABC):
    def __init__(self, resize_min: typing.Union[int, typing.Tuple[int, int]] = 40, to_greyscale: bool = True) -> None:
        super().__init__()
        pipeline = [T.ToTensor(),
                    T.Resize(resize_min),
                    ]
        if to_greyscale:
            pipeline.append(T.Grayscale())

        self._resizer = T.Compose(pipeline)

    @abc.abstractmethod
    def _custom_transform(self, screen_img_array: np.ndarray, env: Env) -> np.ndarray:
        pass

    def convert(self, screen_img_array: np.ndarray, env: Env) -> torch.Tensor:
        screen_img_array = self._custom_transform(screen_img_array, env)

        screen_img_array = np.ascontiguousarray(screen_img_array)
        # screen_img_array = np.ascontiguousarray(screen_img_array, dtype=np.float32) / 255
        # screen = torch.from_numpy(screen_img_array)
        return self._resizer(screen_img_array)


class SimpleConverter(ScreenConverter):

    def _custom_transform(self, screen_img_array: np.ndarray, env: Env) -> np.ndarray:
        return screen_img_array
