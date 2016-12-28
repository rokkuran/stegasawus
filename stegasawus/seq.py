import numpy as np
from functools import partial


def all_the_kings_men(n, **kwargs):
    return xrange(n)


def _gen_skipy(y, n, **kwargs):
    i = 0
    x = 0
    while i <= n:
        x += y
        yield x
        i += 1


def skipy(y):
    return partial(_gen_skipy, y=y)


def _gen_rand_jump(seed, max_jump, n, **kwargs):
    i = 0
    x = 0
    np.random.seed(seed)
    while i <= n:
        x += np.random.randint(1, max_jump)
        yield x
        i += 1


def rand_jump(seed, max_jump):
    return partial(_gen_rand_jump, seed=seed, max_jump=max_jump)


def _gen_rand_jump_circle(seed, max_jump, n, verbose=True, **kwargs):
    np.random.seed(seed)
    i = 0
    x = 0
    modified = []
    n_resets = 0
    while i <= n:
        if x >= n:
            # return to beginning of array once end reached
            x = -1 + np.random.randint(1, max_jump)
            n_resets += 1
            if verbose:
                print 'x reset; n_resets = %d' % n_resets
        else:
            x += np.random.randint(1, max_jump)
            if x not in modified and x < n:
                yield x
                modified.append(x)
                i += 1


def rand_jump_circle(seed, max_jump):
    return partial(_gen_rand_jump_circle, seed=seed, max_jump=max_jump)


def _gen_rand_darts(seed, n):
    np.random.seed(seed)
    i = 0
    remaining = range(n)
    while i <= n:
        x = np.random.randint(0, len(remaining))
        yield remaining[x]
        remaining.pop(x)
        i += 1


def rand_darts(seed):
    return partial(_gen_rand_darts, seed=seed)
