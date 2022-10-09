# DDQN强化学习模型玩Super Mario 

## 1. 简介

本项目的目标是训练一个基于DDQN(Double Deep Q-Networks)的强化学习模型，在仅输入游戏画面的前提下，由模型自主决策通关游戏。
 
本项目改造自PyTorch的教程文档：[https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

Super Mario环境介绍：[https://github.com/Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

## 2. 运行环境准备

- Python版本 >= 3.8 
- 虚拟环境：
  - anaconda、virtualenv皆可
- 环境安装步骤(以windows下的anaconda为例)：
  1. 创建虚拟环境
      ```shell
      conda create -y --name super_mario python=3.10
      ```
  2. 激活虚拟环境
      ```shell
      conda activate super_mario
      ```
  3. 设置pypi镜像源加速pip
      ```shell
      pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
      ```
  4. 安装依赖
      ```shell
      pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113  
      ```

## 3. 模块介绍

- **env_test.ipynb**: 环境演示与基础概念（需在jupyter notebook中打开）
- **train_script.py**: 模型训练主程序脚本
  - 启动训练后在终端中执行命令：``tensorboard --logdir outputs`` 即可启动tensorboard，终端会输出页面地址，复制到浏览器中即可访问。
- **play_script.py**: 模型效果评估脚本