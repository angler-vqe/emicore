from itertools import accumulate
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SingularGramError(RuntimeError):
    pass


def append_train_log(fpath, stats):
    with open(fpath, 'a') as fd:
        # write the header if we are at byte 0 of the file
        if not fd.tell():
            fd.write(','.join(str(key) for key in stats) + '\n')
        # write the supplied payload line
        fd.write(','.join(str(value) for value in stats.values()) + '\n')


def circuit_param_size(circuit, n_layers):
    if circuit == 'generic':
        return 2 + n_layers * 4
    elif circuit == 'esu2':
        return 2 + n_layers * 2
    raise RuntimeError(f'No such circuit: \'{circuit}\'')


def expand_params(angles, n_qbits):
    if len(angles.shape) == 1:
        angles = angles[None]
    if len(angles.shape) == 2:
        shape = angles.shape + (n_qbits,)
        angles = np.repeat(angles, n_qbits, axis=1).reshape(shape)
    elif len(angles.shape) != 3:
        raise TypeError('Parameter values have to have 1, 2 or 3 dimensions.')
    return angles


def state_fidelity(wf1, wf2):
    return np.absolute(np.dot(np.conj(np.transpose(wf1)), wf2))


def grid_search_gamma(model, min_gamma=1.0, max_gamma=30.0, num=20, skip_middle=True, loss='loo'):
    '''Grid search best gamma and change model in-place.'''
    if len(model.x_train) <= 1:
        # cannot do grid search with less than 2 samples
        return model.kernel.gamma
    if skip_middle:
        gammas = np.linspace(min_gamma, max_gamma, num + 1)
        gammas = np.concatenate([gammas[:num // 2], gammas[num // 2 + 1:]])
    else:
        gammas = np.linspace(min_gamma, max_gamma, num)
    # gammas = gammas[(1. < gammas) * (gammas < 10.)]

    lossfn = {
        'loo': model.loocv_mll_closed,
        'mll': model.log_likelihood,
    }[loss]

    # fall back to original gamma if best
    sq_loss = [(lossfn().item(), model.kernel.gamma.item())]
    for gamma in gammas:
        model.kernel.gamma[()] = gamma
        try:
            model.reinit()
        except RuntimeError:
            logging.warning(f'SingularGramError on gamma = {gamma:.3e}. Skipping...')
            continue
        sq_loss.append((lossfn().item(), gamma))

    _, best_gamma = max(sq_loss)
    model.kernel.gamma[()] = best_gamma
    model.reinit()
    return best_gamma


def interval_schedule(param_string):
    '''Create a function from a schedule string in the form of ``[m*]n+[...]``, where * repeats the same number n, m
    times, and each number n specifies after how many steps the returned function should evaluate to True.
    The last number n is repeated forever.
    For example, ``'1+2*2+5'`` evaluates to True at steps ``0, 2, 4, 9, 14, 19, ...``.
    When the evaluated schedule is empty, the returned function always evaluates to ``False``.
    '''
    if not param_string:
        return lambda step: False
    if not isinstance(param_string, str):
        return param_string
    params = [[int(sub) for sub in elem.split('*', maxsplit=1)] for elem in param_string.split('+')]
    schedule = sum([elem if len(elem) == 1 else elem[0] * [elem[1]] for elem in params], [])
    if not schedule:
        return lambda step: False
    final_interval = schedule[-1]
    final_sum = sum(schedule[:-1])
    schedule = set(accumulate(schedule[:-1]))

    def hitstep(step):
        if step + 1 >= final_sum:
            return (step + 1 - final_sum) % final_interval == 0
        return step + 1 in schedule
    return hitstep
