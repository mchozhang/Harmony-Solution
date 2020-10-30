#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from game.game_env import GameEnv
from game.query_game_data import query_level
from game import utils

dir_path = os.path.dirname(os.path.realpath(__file__))


def eval_level(level):
    """
    evaluate the policy of a level, print the sequence of actions and the result.
    :param level: level of the game
    """
    cells = query_level(level)
    size = len(cells)
    env = GameEnv(size, cells)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    time_step = tf_env.reset()
    policy = tf.saved_model.load(os.path.join(dir_path, 'policy_lv{0}'.format(level)))
    step_counter = 0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        print(level, step_counter, env.action_mapper[action_step.action.numpy()[0]])
        time_step = tf_env.step(action_step.action)
        step_counter += 1

    if step_counter == env.solution_length:
        print("win.")
    else:
        print("lost.")


def get_action_from_policy(level, grid):
    size = len(grid)
    env = GameEnv(size, grid)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    time_step = tf_env.reset()
    policy = utils.get_policy(level)
    action_step = policy.action(time_step)
    action = env.action_mapper[action_step.action.numpy()[0]]
    return list(action)


def main():
    utils.init_action_mappers()
    # for i in range(21, 24):
    #     eval_level(i)
    eval_level(30)


if __name__ == "__main__":
    main()
