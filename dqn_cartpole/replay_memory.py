#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple, deque
import random

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
