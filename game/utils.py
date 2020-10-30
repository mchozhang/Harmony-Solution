#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# mapper of the number of possible actions of a size of grid
action_num_mapper = {3: 18, 4: 48, 5: 100}

# mappers of the mapper of possible actions of a size of grid
# example: { 3: { 0: (0,0,0,1), 1: (0, 0, 1, 0) } }
action_mappers = dict()

# mapper of a state and its legal action mask
state_mask_mapper = dict()
state_action_num_mapper = dict()

# action legality mapper,
# where key is action tuple of 6 values, value is tuple of legality and uniqueness
swap_legality_mapper = dict()

# column row legality mapper
row_legality_mapper = dict()
col_legality_mapper = dict()

# trained policies
policy_mapper = dict()


def init():
    init_action_mappers()


def init_action_mappers():
    """
    initial the action_num_mapper and action_mapper,
    should be called before initializing the game environment
    """
    for size in range(3, 7):
        index = 0
        action_mappers[size] = dict()
        for i in range(size):
            for j in range(size):
                # swap in the same column
                for k in range(i + 1, size):
                    action_mappers[size][index] = (i, j, k, j)
                    index += 1
                for k in range(j + 1, size):
                    action_mappers[size][index] = (i, j, i, k)
                    index += 1
        action_num_mapper[size] = index


def get_action_mapper(size: int):
    """
    get the action_mapper of a particular size of grid
    :return: mapper of actions, where the key is an integer index and
             value is the tuple of action
    """
    return action_mappers[size]


def get_action(index: int, size: int):
    """
    get an action by the index of the action_mapper and the size of the grid,
    :return: an action is a tuple of 4 values indicating 2 swapping positions
             example: (0, 0, 0, 1)
    """
    return action_mappers[size][index]


def get_action_num(size: int) -> int:
    """
    get number of possible actions of a size of grid
    :return: number of possible actions
    """
    return action_num_mapper[size]


def get_mask_from_buffer(state_bytes: bytes):
    """
    get legal action mask of a state
    """
    result = state_mask_mapper.get(state_bytes, None)
    if result is None:
        return None, 0
    else:
        mask = np.frombuffer(result, dtype=np.int32)
    return mask, state_action_num_mapper.get(state_bytes, 0)


def add_state_mask_to_buffer(state_bytes: bytes, mask: np.ndarray, action_num: int):
    """
    add state mask map to buffer
    """
    state_mask_mapper[state_bytes] = mask.tostring()
    state_action_num_mapper[state_bytes] = action_num


def is_legal_swap(step1, step2, target1, target2, row1, row2):
    """
    check legality and uniqueness of an action,
    an action is unique when the swap in the same column makes both cells complete
    """
    group = (step1, step2, target1, target2, row1, row2)
    if group in swap_legality_mapper:
        return swap_legality_mapper[group]
    else:
        complete1 = step1 == 1 and target1 == row2
        complete2 = step2 == 1 and target2 == row1
        uniqueness = row1 != row2 and complete1 and complete2
        if uniqueness:
            legality = True
        else:
            legality = (step1 > 1 or complete1) and (step2 > 1 or complete2)
        swap_legality_mapper[group] = legality, uniqueness
        return legality, uniqueness


def column_check(key):
    return col_legality_mapper.get(key, None)


def add_column_legality(key, legality):
    col_legality_mapper[key] = legality


def row_check(key):
    return row_legality_mapper.get(key, None)


def add_row_legality(key, legality):
    row_legality_mapper[key] = legality


def hash_array(array: np.ndarray):
    return array.tobytes()


def load_trained_policies():
    """
    pre-load the trained policies
    """
    for i in range(1, 31):
        try:
            policy_mapper[i] = tf.saved_model.load(os.path.join(dir_path, 'trained_policies/policy_lv{0}'.format(i)))
        except Exception:
            pass


def get_policy(level):
    return policy_mapper.get(level)
