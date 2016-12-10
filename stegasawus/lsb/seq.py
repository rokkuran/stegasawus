import numpy as np
from functools import partial


def all_the_kings_men(n):
    return xrange(n)


def _gen_skipy(y, n):
    i = 0
    x = 0
    while i <= n:
        x += y
        yield x
        i += 1


def skipy(y):
    return partial(_gen_skipy, y=y)


def _gen_skip_rand(seed, max_skip, n):
    i = 0
    x = 0
    np.random.seed(seed)
    while i <= n:
        x += np.random.randint(1, max_skip)
        yield x
        i += 1


def skip_rand(seed, max_skip):
    return partial(_gen_skip_rand, seed=seed, max_skip=max_skip)


def fibonacci(n):
    a, b = 0, 1
    i = 0
    while i <= n:
        yield a
        a, b = b, a + b
        i += 1
