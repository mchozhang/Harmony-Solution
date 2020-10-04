#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from game import utils
import time


class GameEnv(py_environment.PyEnvironment):
    """
    python environment of the game
    """
    time_1 = 0
    time_2 = 0
    time_3 = 0

    def timeit(method):
        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            GameEnv.time_1 += time.time() - start
            return result

        return timed

    def timeit_2(method):
        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            GameEnv.time_2 += time.time() - start
            return result

        return timed

    def timeit_3(method):
        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            GameEnv.time_3 += time.time() - start
            return result

        return timed

    STEP = 0
    TARGET = 1

    # state of the game
    STATE_WIN = 0
    STATE_LOSS = 1
    STATE_ONGOING = 2

    # reward
    REWARD_LOSS = -10
    REWARD_WIN = 100
    REWARD_ILLEGAL_ACTION = -10
    REWARD_COMPLETE_CELL = 0.02
    REWARD_PLACED_CELL = 0.01
    REWARD_NORMAL_STEP = 0.01

    def __init__(self, size, grid=None):
        """
        initialize the python environment of the game
        :param grid: 2d list of cell json object
        """
        super(GameEnv, self).__init__()
        self._discount = 1
        self.size = size
        self.action_num = utils.get_action_num(self.size)
        self.action_mapper = utils.get_action_mapper(self.size)

        if grid is not None:
            self.set_grid(grid)

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
        self._observation_spec = obs_spec, mask_spec
        self._game_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def set_grid(self, grid):
        self.initial_state = GameEnv.grid_to_array(grid)
        self.cell_num = self.size ** 2
        self._states = GameEnv.copy_array(self.initial_state)
        self._initial_complete_cell_num = np.count_nonzero(self._states[:, :, GameEnv.STEP] == 0)
        self._initial_placed_cell_num = len([1 for i in range(self.size) for j in range(self.size)
                                             if self._states[i, j, GameEnv.STEP] < 3 and
                                             self._states[i, j, GameEnv.TARGET] == i])
        self._complete_cell_num = self._initial_complete_cell_num
        self._placed_cell_num = self._initial_placed_cell_num
        step_sum = self._states[:, :, GameEnv.STEP].sum()
        self.solution_length = step_sum // 2
        if step_sum % 2 == 1:
            raise ValueError('Wrong grid data.')
        self._update_legal_actions()

    def _get_observation(self):
        return GameEnv.copy_array(self._states), self._legal_action_mask

    def _update_legal_actions(self):
        """
        update the mask and number of current legal actions
        """
        # get results from buffer
        key = utils.hash_array(self._states)
        mask, legal_action_num = utils.get_mask_from_buffer(key)
        if mask is not None:
            self._legal_action_mask = mask
            self._legal_action_num = legal_action_num
            return

        self._legal_action_mask = np.zeros((self.action_num,), dtype=np.int32)
        self._legal_action_num = 0
        for i, action in self.action_mapper.items():
            row1, col1, row2, col2 = action
            grid = GameEnv.copy_array(self._states)
            step1, step2 = grid[row1, col1, GameEnv.STEP], grid[row2, col2, GameEnv.STEP]
            target1, target2 = grid[row1, col1, GameEnv.TARGET], grid[row2, col2, GameEnv.TARGET]

            legality, uniqueness = GameEnv.exchangeable(step1, step2, target1, target2, row1, row2)
            # find if there is a unique action
            if uniqueness:
                self._legal_action_mask = np.zeros((self.action_num,), dtype=np.int32)
                self._legal_action_mask[i] = 1
                self._legal_action_num = 1
                break

            # swap if causing no immediate dead cells
            if legality:
                grid[row1, col1, GameEnv.STEP], grid[row2, col2, GameEnv.STEP] = step2 - 1, step1 - 1
                grid[row1, col1, GameEnv.TARGET], grid[row2, col2, GameEnv.TARGET] = target2, target1
            else:
                continue

            # swap in the same row
            if row1 == row2 \
                    and GameEnv.row_check(grid, row1) \
                    and GameEnv.column_check(grid, col1) \
                    and GameEnv.column_check(grid, col2):
                self._legal_action_mask[i] = 1
                self._legal_action_num += 1
            # swap in the same column
            elif GameEnv.row_check(grid, row1) \
                    and GameEnv.row_check(grid, row2) \
                    and GameEnv.column_check(grid, col1):
                self._legal_action_mask[i] = 1
                self._legal_action_num += 1

        # add result to buffer
        utils.add_state_mask_to_buffer(key, self._legal_action_mask, self._legal_action_num)

    def _update_complete_state(self, row1, col1, row2, col2):
        """
        update complete cell num
        """
        step1, step2 = self._states[row1, col1, GameEnv.STEP], self._states[row2, col2, GameEnv.STEP]
        target1, target2 = self._states[row1, col1, GameEnv.TARGET], self._states[row2, col2, GameEnv.TARGET]
        if step1 == 0:
            self._complete_cell_num += 1

        if step2 == 0:
            self._complete_cell_num += 1

        if target1 == row1 and target2 != row1 and step1 < 3:
            self._placed_cell_num += 1

        if target2 == row2 and target1 != row2 and step2 < 3:
            self._placed_cell_num += 1

    def _reset(self) -> ts.TimeStep:
        """
        reset the environment to the initial state, return the first TimeStep
        :return: TimeStep
        """
        self._game_ended = False
        self._states = GameEnv.copy_array(self.initial_state)
        self._update_legal_actions()
        self._complete_cell_num = self._initial_complete_cell_num
        self._placed_cell_num = self._initial_placed_cell_num
        return ts.restart(self._get_observation())

    def _step(self, action_index) -> ts.TimeStep:
        """
        the environment steps forward by taking the action
        :param action_index: index of the action in the action mapper
        :return: TimeStep object after taking the action
        """
        if self._game_ended:
            return self.reset()

        # illegal action terminates the game, todo: remove
        if self._legal_action_mask[action_index] == 0:
            self._game_ended = True
            return ts.termination(self._get_observation(), GameEnv.REWARD_ILLEGAL_ACTION)

        # update grid data after the swap
        row1, col1, row2, col2 = self.action_mapper[action_index.item()]
        self._states[row1, col1, GameEnv.STEP], self._states[row2, col2, GameEnv.STEP] = \
            self._states[row2, col2, GameEnv.STEP] - 1, self._states[row1, col1, GameEnv.STEP] - 1
        self._states[row1, col1, GameEnv.TARGET], self._states[row2, col2, GameEnv.TARGET] = \
            self._states[row2, col2, GameEnv.TARGET], self._states[row1, col1, GameEnv.TARGET]

        self._update_legal_actions()
        self._update_complete_state(row1, col1, row2, col2)

        # check game state and reward
        state, reward = self._check_state()
        if state != GameEnv.STATE_ONGOING:
            self._game_ended = True
            return ts.termination(self._get_observation(), reward)

        return ts.transition(self._get_observation(), reward, discount=self._discount)

    def _check_state(self):
        """
        pre-condition: game is not dead
        check if the given states are final and calculate reward
        :return: a tuple of game state(win, loss, not final) and reward
        """
        if self._complete_cell_num == self.cell_num:
            return GameEnv.STATE_WIN, GameEnv.REWARD_WIN
        elif self._legal_action_num == 0:
            return GameEnv.STATE_LOSS, GameEnv.REWARD_LOSS
        else:
            # game is not finished
            reward = self._complete_cell_num * GameEnv.REWARD_COMPLETE_CELL + \
                     (self._placed_cell_num - self._complete_cell_num) * GameEnv.REWARD_PLACED_CELL \
                if self._placed_cell_num > self.cell_num * 0.3 else 0
            # reward = self._complete_cell_num * GameEnv.REWARD_COMPLETE_CELL \
            #     if self._complete_cell_num > self.cell_num * 0.3 else 0
            return GameEnv.STATE_ONGOING, reward

    @staticmethod
    def exchangeable(step1, step2, target1, target2, row1, row2):
        """
        whether 2 cells can swap without causing immediate death
        """
        return utils.is_legal_swap(step1, step2, target1, target2, row1, row2)

    @staticmethod
    def column_check(grid, col_index):
        """
        pre-condition: no immediate dead cell in the grid
        check whether a column is alive
        """
        col = grid[:, col_index]
        key = utils.hash_array(col)
        legality = utils.column_check(key)
        if legality is not None:
            return legality

        targeted = set()
        # no 2 cells targeting at the same place
        for i, cell in enumerate(col):
            if cell[GameEnv.STEP] == 0:
                if i in targeted:
                    utils.add_column_legality(key, False)
                    return False
                targeted.add(i)
            elif cell[GameEnv.STEP] == 1:
                if cell[GameEnv.TARGET] in targeted:
                    utils.add_column_legality(key, False)
                    return False
                targeted.add(cell[GameEnv.TARGET])

        utils.add_column_legality(key, True)
        return True

    @staticmethod
    def row_check(grid, row_index):
        row = grid[row_index, :]
        key = utils.hash_array(row)
        legality = utils.row_check(key)
        if legality is not None:
            return legality

        # whether all cells are in the correct row
        if np.count_nonzero(row[:, GameEnv.TARGET] == row_index) != len(row):
            utils.add_row_legality(key, True)
            return True
        else:
            # count cell with 0, 1 and 2 steps remained
            unique, counts = np.unique(row[:, GameEnv.STEP], return_counts=True)
            counter = dict(zip(unique, counts))
            count_0, count_1, count_2 = counter.get(0, 0), counter.get(1, 0), counter.get(2, 0)
            incomplete_num = count_0 + count_1 + count_2
            incomplete_steps = count_1 + count_2 * 2

            # if all cells have 2 steps remained at best,
            # row is dead if odd number of steps remained
            legality = incomplete_num == len(row) and incomplete_steps % 2 == 0
            utils.add_row_legality(key, legality)
            return legality

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
        return observation

    @staticmethod
    def copy_array(array):
        return np.array(array)
