#!/usr/bin/env python
# -*- coding: utf-8 -*-

action_num_mapper = {3: 18, 4: 48, 5: 100}

action_mappers = dict()


def init_action_mappers():
    for size in range(3, 6):
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


def get_action_mapper(size=4):
    return action_mappers[size]


def get_action(index, size=4):
    return action_mappers[size][index]


def get_action_num(size=4):
    return action_num_mapper[size]
