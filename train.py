#!/usr/bin/env python
# Run the train function in the game module
# to train a specified level or a range of levels
import time
import sys
from game.train import train_level


def main():
    start = time.time()
    if len(sys.argv) == 2:
        level = int(sys.argv[1])
        train_level(level, 2, True, max_iterations=1000000)

    elif len(sys.argv) == 3:
        start_level = int(sys.argv[1])
        end_level = int(sys.argv[2])
        for i in range(start_level, end_level + 1):
            train_level(i, 3,
                        collect_random_steps=True,
                        max_iterations=50000)
    else:
        print('Wrong arguments.')
    print('time:', time.time() - start)


if __name__ == '__main__':
    main()
