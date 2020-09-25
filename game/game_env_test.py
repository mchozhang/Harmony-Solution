#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
from game.game_env import GameEnv
import game.utils
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import test_utils
from tf_agents.environments import utils as env_utils
from tf_agents.environments import tf_py_environment


class GameEnvironmentTest(test_utils.TestCase):
    def setUp(self):
        super(GameEnvironmentTest, self).setUp()
        # level 10
        cells = [
            [
            {
                "targetRow": 0,
                "steps": 1,
                "col": 0,
                "row": 0
            },
            {
                "targetRow": 0,
                "steps": 1,
                "col": 1,
                "row": 0
            },
            {
                "targetRow": 2,
                "steps": 1,
                "col": 2,
                "row": 0
            }
        ],
            [
                {
                    "targetRow": 1,
                    "steps": 3,
                    "col": 0,
                    "row": 1
                },
                {
                    "targetRow": 1,
                    "steps": 3,
                    "col": 1,
                    "row": 1
                },
                {
                    "targetRow": 1,
                    "steps": 2,
                    "col": 2,
                    "row": 1
                }
            ],
            [
                {
                    "targetRow": 2,
                    "steps": 1,
                    "col": 0,
                    "row": 2
                },
                {
                    "targetRow": 2,
                    "steps": 1,
                    "col": 1,
                    "row": 2
                },
                {
                    "targetRow": 0,
                    "steps": 1,
                    "col": 2,
                    "row": 2
                }
            ]
        ]
        np.random.seed(0)
        game.utils.init_action_mappers()
        self.discount = np.asarray(1., dtype=np.float32)
        self.env = GameEnv(cells)
        ts = self.env.reset()

    def test_validate_specs(self):
        env_utils.validate_py_environment(self.env, episodes=10)

    def test_validate_mask(self):
        env = tf_py_environment.TFPyEnvironment(self.env)
        policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=env.time_step_spec(),
            action_spec=env.action_spec(),
            observation_and_action_constraint_splitter=GameEnv.obs_and_mask_splitter)

        driver = dynamic_step_driver.DynamicStepDriver(env, policy, num_steps=1)

        for i in range(10):
            time_step, _ = driver.run()
            action_step = policy.action(time_step)
            print(game.utils.get_action(action_step.action.numpy()[0], 3))


if __name__ == '__main__':
    test_utils.main()
