#!/usr/bin/env python
# -*- coding: utf-8 -*-
from game.game_env import GameEnv
from game.query_game_data import query_level
from game.mask_q_network import MaskedQNetwork
import game.utils
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import q_network
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_saver
from tf_agents.utils import common


class Agent:
    def __init__(self, size):
        env = GameEnv(size)
        env = tf_py_environment.TFPyEnvironment(env)
