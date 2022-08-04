#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision.transforms as T
import abc
from gym import Env

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class ScreenConverter(abc.ABC):
    def __init__(self, resize_min: int = 40, to_greyscale: bool = False) -> None:
        super().__init__()
        pipeline = [T.ToPILImage(),
                    T.Resize(resize_min, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor()]
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
        # Resize, and add a batch dimension (BCHW)
        return self._resizer(screen_img_array).unsqueeze(0)


class CartPoleScreenConverter(ScreenConverter):

    def __init__(self, resize_min: int = 40, to_greyscale: bool = False, center_cart: bool = True) -> None:
        super().__init__(resize_min, to_greyscale)
        self._center_cart = center_cart

    @classmethod
    def _get_cart_location(cls, screen_width: int, env: Env):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def _custom_transform(self, screen_img_array: np.ndarray, env: Env) -> np.ndarray:
        # Shape: H, W, C
        screen_height, screen_width, _ = screen_img_array.shape
        screen_img_array = screen_img_array[int(screen_height * 0.4): int(screen_height * 0.8)]

        if self._center_cart:
            view_width = int(screen_width * 0.6)
            cart_position = self._get_cart_location(screen_width=screen_width, env=env)

            if cart_position < view_width // 2:
                slice_range = slice(view_width)
            elif cart_position > screen_width - view_width // 2:
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_position - view_width // 2, cart_position + view_width // 2)

            screen_img_array = screen_img_array[:, slice_range]

        return screen_img_array
