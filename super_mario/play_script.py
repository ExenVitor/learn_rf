#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from screen_convert import SimpleConverter
from state_gen import StackStateGenerator
from env_factory import EnvFactory
from eval import Evaluator
from trainer import CheckpointManager
from agent import DDQNAgent
from train_script import OUTPUT_BASE_DIR, CUSTOM_MOVEMENTS

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


def main(model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_factory = EnvFactory(env_id="SuperMarioBros-1-1-v0",
                             movements=CUSTOM_MOVEMENTS,
                             skip_frames=4,
                             num_stack=4)

    screen_convert = SimpleConverter(resize_min=(84, 84))
    state_generator = StackStateGenerator(screen_converter=screen_convert)

    checkpoint_manager = CheckpointManager(base_output_dir=OUTPUT_BASE_DIR, model_id=model_id)

    agent = DDQNAgent.load(save_path=checkpoint_manager.get_best_checkpoint_path(), device=device)

    eval_env = env_factory.gen_env(random_seed=84)
    evaluator = Evaluator(env=eval_env,
                          state_generator=state_generator,
                          device=device,
                          agent=agent,
                          video_record_dir=os.path.join(checkpoint_manager._model_path, "eval_video"))

    eval_result = evaluator.run(render=True)

    print(eval_result.stat_dict)


if __name__ == '__main__':
    main(model_id="20220812_191147_SuperMario_DDQN")
