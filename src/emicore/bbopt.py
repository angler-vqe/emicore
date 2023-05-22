import math

import numpy as np
from noisyopt import minimizeSPSA
from scipy.optimize import minimize


_bbopts = {}


def register_bbopt(name):
    def wrapped(func):
        _bbopts[name] = func
        return func
    return wrapped


@register_bbopt('spsa')
def spsa(energy, vector, n_iter=None):
    if n_iter is None:
        n_iter = 500000
    vec_bound = np.empty((len(vector), 2), dtype=object)
    vec_bound[:] = np.array([0, 2 * math.pi])
    return minimizeSPSA(energy, vector, bounds=vec_bound, paired=False, niter=n_iter)


@register_bbopt('nelder_mead')
def nelder_mead(energy, vector, n_iter=None):
    if n_iter is None:
        n_iter = 200 * len(vector)
    return minimize(energy, vector, method="Nelder-Mead", options={'maxiter': n_iter})
