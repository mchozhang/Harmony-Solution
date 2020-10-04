#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the policy of a specific level
"""
import sys
from game.eval_policy import eval_level
from game import utils


def main():
    utils.init_action_mappers()
    level = int(sys.argv[1])
    eval_level(level)


if __name__ == '__main__':
    main()
