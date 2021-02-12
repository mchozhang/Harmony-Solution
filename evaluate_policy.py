#!/usr/bin/env python
# -*- coding: utf-8 -*-
# evaluate the policy of a level
# argv: level

import sys
from game.eval_policy import eval_level


def main():
    level = sys.argv[1]
    eval_level(level)


if __name__ == "__main__":
    main()
