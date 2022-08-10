#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from screen_convert import SimpleConverter
from state_gen import StackStateGenerator
from trainer import DQNTrainer
from env_factory import EnvFactory
from eval import Evaluator

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"

CUSTOM_MOVEMENTS = [
    ['right'],
    ['right', 'A']
]

OUTPUT_BASE_DIR = "outputs"
NUM_EPISODES = 4000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_factory = EnvFactory(env_id="SuperMarioBros-1-1-v0",
                             movements=CUSTOM_MOVEMENTS,
                             skip_frames=4,
                             num_stack=4)

    env = env_factory.gen_env(random_seed=42)

    screen_convert = SimpleConverter(resize_min=(84, 84))
    state_generator = StackStateGenerator(screen_converter=screen_convert)

    trainer = DQNTrainer(env=env,
                         state_generator=state_generator,
                         device=device,
                         base_output_dir=OUTPUT_BASE_DIR,
                         model_tag="SuperMario_DDQN")

    trainer.train(num_episodes=NUM_EPISODES, episode_end_callback=episode_end_callback)

    eval_env = env_factory.gen_env(random_seed=84)
    evaluator = Evaluator(env=eval_env,
                          state_generator=state_generator,
                          device=device,
                          agent=trainer.agent)

    eval_result = evaluator.run(render=True)
    eval_result.save(trainer.model_output_dir)


def episode_end_callback(cur_episode: int, cur_episode_duration: int, total_steps: int):
    print(f"Cur EP: {cur_episode}, Duration: {cur_episode_duration}, Total Steps: {total_steps}")


if __name__ == '__main__':
    main()
