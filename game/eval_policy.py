#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from game.game_env import GameEnv
from game.query_game_data import query_level
from game import utils


def eval_level(level):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    cells = query_level(level)
    size = len(cells)
    env = GameEnv(cells)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    time_step = tf_env.reset()
    policy = tf.saved_model.load(os.path.join(dir_path, 'policy_{0}x{0}'.format(size)))
    step_counter = 0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        print(level, step_counter, env.action_mapper[action_step.action.numpy()[0]])
        time_step = tf_env.step(action_step.action)
        step_counter += 1


def main():
    # for i in range(11, 15):
    #     eval_level(i)
    utils.init_action_mappers()
    eval_level(23)


if __name__ == "__main__":
    main()
