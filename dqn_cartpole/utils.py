#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


def conv2d_size_out(size: int, kernel_size: int, stride: int = 1, padding: int = 0):
    """
    compute the output size of Conv2d layer
    """
    return (size - kernel_size + 2 * padding) // stride + 1


def check_and_mkdirs(dir_path):
    if dir_path is None:
        return
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
