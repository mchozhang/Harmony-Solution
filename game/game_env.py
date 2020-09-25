#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from game.utils import get_action_num, get_action, get_action_mapper


class GameEnv(py_environment.PyEnvironment):
    """
    python environment of the game
    """
    STEP = 0
    TARGET = 1

    # state of the game
    STATE_WIN = 0
    STATE_LOSS = 1
    STATE_NOT_FINAL = 2

    # reward
    REWARD_LOSS = -10
    REWARD_WIN = 100
    REWARD_ILLEGAL_ACTION = -10
    REWARD_COMPLETE_CELL = 0.01
    REWARD_NORMAL_STEP = 0.01

    def __init__(self, grid):
        """
        initialize the python environment of the game
        :param grid: 2d list of cell json object
        """
        super(GameEnv, self).__init__()
        self.size = len(grid)
        self.cell_num = self.size ** 2
        self.grid = grid
        self._discount = 0.99
        self._states = GameEnv.grid_to_array(grid)
        self.solution_length = self._states[:, :, GameEnv.STEP].sum() // 2

        self.action_num = get_action_num(self.size)
        self.action_mapper = get_action_mapper(self.size)
        self._update_legal_actions()

        # action is key of the action mapper
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.action_num - 1, name='action')

        # observation specification is the N * N grid with STEP and TARGET values for each cell
        mask_spec = array_spec.ArraySpec(shape=(self.action_num,), dtype=np.int32, name='mask')
        obs_spec = array_spec.BoundedArraySpec(
            shape=(self.size, self.size, 2),
            dtype=np.int32,
            minimum=0,
            maximum=self.size,
            name='observation')
        self._observation_spec = {'state': obs_spec, 'mask': mask_spec}
        self._game_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

        # return self._observation_spec['state']

    def _get_observation(self):
        return {
            'state': np.array(self._states, dtype=np.int32),
            'mask': self._legal_action_mask(),
        }

    def _legal_action_mask(self):
        """
        boolean array that mask out illegal actions
        """
        mask = np.zeros((self.action_num,), dtype=np.int32)
        for i in self._current_legal_actions:
            mask[i] = 1
        return mask

    def _update_legal_actions(self):
        """
        update current legal actions
        """
        self._current_legal_actions = [i for i, v in self.action_mapper.items()
                                       if GameEnv.exchangeable(self._states, v[0], v[1], v[2], v[3])]

    def _reset(self) -> ts.TimeStep:
        """
        reset the environment to the initial state, return the first TimeStep
        :return: TimeStep
        """
        self._game_ended = False
        self._states = GameEnv.grid_to_array(self.grid)
        self._update_legal_actions()
        return ts.restart(self._get_observation())

    def _step(self, action_index) -> ts.TimeStep:
        """
        the environment steps forward by taking the action
        :param action_index: index of the action in the action mapper
        :return: TimeStep object after taking the action
        """
        if self._game_ended:
            return self.reset()

        # the first and the second swapping cell
        row, col, row2, col2 = self.action_mapper[action_index.item()]
        first = self._states[row, col, :]
        second = self._states[row2, col2, :]

        # illegal action terminates the game
        if first[GameEnv.STEP] == 0 or second[GameEnv.STEP] == 0 or not (row == row2 or col == col2):
            self._game_ended = True
            return ts.termination(self._get_observation(), GameEnv.REWARD_ILLEGAL_ACTION)

        # update grid data after the swap
        first[GameEnv.STEP], second[GameEnv.STEP] = second[GameEnv.STEP] - 1, first[GameEnv.STEP] - 1
        first[GameEnv.TARGET], second[GameEnv.TARGET] = second[GameEnv.TARGET], first[GameEnv.TARGET]

        state, reward = self._check_state()
        if state != GameEnv.STATE_NOT_FINAL:
            self._game_ended = True
            return ts.termination(self._get_observation(), reward)

        return ts.transition(self._get_observation(), reward, discount=self._discount)

    def _check_state(self):
        """
        check if the given states are final and calculate reward
        :return: a tuple of game state(win, loss, not final) and reward
        """
        # update current legal action
        self._update_legal_actions()

        # record the targeted position, the game is dead if
        # if more than 1 cell targeting to the same position
        targeted = set()

        # count the number of complete cells
        complete_count = 0

        # count the number of complete row
        complete_row = 0

        for i in range(self.size):
            complete_cell_in_row = 0

            # scan cells of each row
            for j in range(self.size):
                cell = self._states[i, j, :]
                if cell[GameEnv.STEP] == 0:
                    # unmovable cell
                    # whether the cell is complete
                    if i == cell[GameEnv.TARGET]:
                        complete_cell_in_row += 1
                    else:
                        return GameEnv.STATE_LOSS, GameEnv.REWARD_LOSS
                    # whether the cell is targeted by other cells
                    if (i, j) in targeted:
                        return GameEnv.STATE_LOSS, GameEnv.REWARD_LOSS
                    else:
                        targeted.add((i, j))
                elif cell[GameEnv.STEP] == 1 and i != cell[GameEnv.TARGET]:
                    # cell that have 1 step remained and not in the right row

                    # whether target to the same cell with other cells
                    if (cell[GameEnv.TARGET], j) in targeted:
                        return GameEnv.STATE_LOSS, GameEnv.REWARD_LOSS
                    else:
                        targeted.add((cell[GameEnv.TARGET], j))

            if complete_cell_in_row == self.size:
                complete_row += 1
            complete_count += complete_cell_in_row

        if complete_row == self.size:
            return GameEnv.STATE_WIN, GameEnv.REWARD_WIN
        else:
            # no legal action
            if len(self._current_legal_actions) == 0:
                return GameEnv.STATE_LOSS, GameEnv.REWARD_LOSS

            # game is not finished
            reward = complete_count * GameEnv.REWARD_COMPLETE_CELL if complete_count > self.cell_num * 0.3 else 0
            # reward = complete_row * GameEnv.REWARD_COMPLETE_CELL
            return GameEnv.STATE_NOT_FINAL, reward

    @staticmethod
    def exchangeable(grid, row, col, row2, col2):
        first = grid[row, col, :]
        second = grid[row2, col2, :]
        return first[GameEnv.STEP] > 0 and second[GameEnv.STEP] > 0 \
               and not (first[GameEnv.STEP] == 1 and first[GameEnv.TARGET] != row2) \
               and not (second[GameEnv.STEP] == 1 and second[GameEnv.TARGET] != row)

    @staticmethod
    def grid_to_array(grid) -> np.ndarray:
        """
        convert a 2d grid to np.array object
        :param grid: 2d list of the grid
        :return: np.array
        """
        return np.array([[[cell["steps"], cell["targetRow"]]
                          for cell in row] for row in grid],
                        dtype=np.int32)

    @staticmethod
    def obs_and_mask_splitter(observation):
        return observation['state'], observation['mask']
