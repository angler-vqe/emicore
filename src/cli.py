import hashlib
import os
import pickle
import re
from collections import namedtuple
from argparse import Namespace
from typing import NamedTuple
from functools import wraps, partial

import click
import h5py
import numpy as np
import torch

from src.energy import BACKENDS
from src.bayesopt.gp import KERNELS
from src.bayesopt.bo import OneShotOptimizer, GradientDescentOptimizer, LBFGSOptimizer, TorchLBFGSOptimizer
from src.bayesopt.bo import SMOOptimizer, EILVSOptimizer, EMICOREOptimizer
from src.bayesopt.bo import ExpectedImprovement, WeightedExpectedImprovement, LowerConfidenceBound, AdaptiveLCB


class FinalProperties:
    def __init_subclass__(cls, *args, **kwargs):
        updates = []
        for key, value in cls.__dict__.items():
            if key[:2] != '__' and callable(value) and hasattr(value, '_final_properties'):
                updates += [(name, property(partial(value, key=name))) for name in value._final_properties]

        for name, prop in updates:
            setattr(cls, name, prop)

    def __init__(self):
        self._dict = {}


def final_property(func):
    @property
    @wraps(func)
    def wrapped(self):
        try:
            return self._dict[func.__name__]
        except KeyError:
            result = func(self)
            self._dict[func.__name__] = result
            return result
    return wrapped


def final_properties(*names):
    def wrapping(func):
        @wraps(func)
        def wrapped(self, key):
            try:
                return self._dict[key]
            except KeyError:
                result = func(self)
                self._dict.update(zip(names, result))
                return self._dict[key]

        wrapped._final_properties = names
        return wrapped
    return wrapping


def csobj(dtype, sep=',', maxsplit=-1, length=-1):
    def wrapped(string):
        if isinstance(string, tuple):
            return string
        result = [dtype(elem) for elem in string.split(sep, maxsplit) if elem]
        if length > 0 and len(result) != length:
            raise RuntimeError(f'Invalid number of fields. Provided {len(result)} but expected {length}!')
        return result
    return wrapped


def option_dict(string):
    if isinstance(string, dict):
        return string
    return dict([elem.split('=', 1) for elem in string.split(',') if elem])


def _append_param(func, param):
    if isinstance(func, click.Command):
        func.params.append(param)
    else:
        if not hasattr(func, '__click_params__'):
            func.__click_params__ = []
        func.__click_params__.append(param)


def namedtuple_as_dict(input):
    if isinstance(input, tuple) and hasattr(input, '_asdict'):
        return namedtuple_as_dict(input._asdict())
    if isinstance(input, Namespace):
        return namedtuple_as_dict(vars(Namespace))
    if isinstance(input, dict):
        return {key: namedtuple_as_dict(value) for key, value in input.items()}
    if isinstance(input, (tuple, list)):
        return type(input)(namedtuple_as_dict(value) for value in input)
    return input


def arrhash(*args):
    hasher = hashlib.sha256()
    for arg in args:
        if isinstance(arg, (np.ndarray, torch.Tensor)):
            arr = np.array(arg)
            flag = arr.flags.writeable
            arr.flags.writeable = False
            hasher.update(arr.data)
            arr.flags.writeable = flag
        else:
            hasher.update(pickle.dumps(arg))
    return hasher.hexdigest()


def arrcache(fname, func, identifiers, keys='value'):
    if fname is None:
        return func()
    single = isinstance(keys, str)
    if single:
        keys = (keys,)

    identifier = arrhash(*identifiers)
    results = None

    if os.path.exists(fname):
        with h5py.File(fname, 'r') as fd:
            if identifier in fd:
                results = tuple(
                    torch.from_numpy(dset[()]) if dset.attrs.get('type', 'numpy') == 'torch' else dset[()]
                    for dset in (fd[f'{identifier}/{key}'] for key in keys)
                )

    if results is None:
        results = func()
        if single:
            results = (results,)
        try:
            with h5py.File(fname, 'a') as fd:
                for key, result in zip(keys, results):
                    fd[f'{identifier}/{key}'] = result
                    fd[f'{identifier}/{key}'].attrs['type'] = 'torch' if isinstance(result, torch.Tensor) else 'numpy'
        except OSError as error:
            raise RuntimeError(f'Unable to cache key \'{identifier}\'.') from error

    if single:
        results, = results
    return results


class Data(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


class TrueSolution(NamedTuple):
    e0: torch.Tensor
    e1: torch.Tensor
    wf: torch.Tensor


class PositiveFloatParam(click.ParamType):
    name = 'positive_float'

    def convert(self, value, param, ctx):
        if not isinstance(value, float):
            value = float(value)
        if value <= 0.:
            self.fail(f'Value {value} is non-positive!', param, ctx)
        return value


PositiveFloat = PositiveFloatParam()


class OptionParams(click.ParamType):
    name = 'OptionParams'
    _rexp = re.compile(
        r'(?P<key>[^\s#=]+)=(?P<value>[^#,\n]+)[,\n]?|'
        r'(?P<comment>#[^\n]*$)|'
        r'(?P<whitespace>\s+)|'
        r'(?P<error>.+)',
        re.MULTILINE
    )

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        cls._types = {
            key: value
            for key, value in cls.__annotations__.items()
            if key not in click.ParamType.__annotations__
        }
        cls._defaults = {}
        cls._help = {}
        for key in cls._types:
            obj = getattr(cls, key, None)
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], str):
                cls._defaults[key] = obj[0]
                cls._help[key] = obj[1]
            else:
                cls._defaults[key] = obj

        cls._return_type = namedtuple(
            cls.__name__,
            list(cls.__annotations__),
            defaults=[cls._defaults[key] for key in cls.__annotations__]
        )

    @classmethod
    def _parse(cls, string):
        for mt in cls._rexp.finditer(string):
            if mt['error'] is not None:
                raise RuntimeError(f'Parsing options failed at \'{mt["error"]}\'!')
            if mt.lastgroup in ('key', 'value'):
                yield (mt['key'], mt['value'])

    @classmethod
    def _members(cls):
        for key, dtype in cls._types.items():
            yield (key, dtype, cls._help.get(key, ''), cls._defaults.get(key, None))

    @classmethod
    def options(cls, prefix='', namespace=None, help=None):
        prefix = prefix.replace('_', '-')

        def decorator(func):
            for key, dtype, dhelp, default in cls._members():
                key = key.replace('_', '-')
                param = click.Option(
                    [f'--{prefix}-{key}' if prefix else f'--{key}'],
                    type=dtype,
                    help=dhelp,
                    default=default,
                    callback=cls._callback(namespace),
                    expose_value=False
                )
                _append_param(func, param)
            return func
        return decorator

    @classmethod
    def _callback(cls, namespace=None):
        def callback(ctx, param, value):
            key = param.name.replace('-', '_')
            if namespace:
                setattr(ctx.params.setdefault(namespace, Namespace()), key, value)
            else:
                ctx.params[key] = value
            return value
        return callback

    def convert(self, value, param, ctx):
        retval = {}
        if isinstance(value, str):
            for key, val in self._parse(value):
                key = key.replace('-', '_')
                if key not in self._types:
                    raise RuntimeError(f'No such option for {param.name}: {key}!')
                retval[key] = self._types[key](val)
        if isinstance(value, self._return_type):
            return value

        return self._return_type(**retval)

    def __repr__(self):
        members = {
            key.replace('_', '-'): (getattr(dtype, '__name__', str(dtype)), self._defaults.get(key, None))
            for key, dtype in self._types.items()
        }
        msg = ', '.join(sorted([
            f'{key}:{dtype}={default}' for key, (dtype, default) in members.items()
        ]))

        return f'{type(self).__name__}({msg})'

    def get_metavar(self, param):
        members = {
            key.replace('_', '-'): (
                getattr(dtype, '__name__', str(dtype)), self._defaults.get(key, None), self._help.get(key, '')
            )
            for key, dtype in self._types.items()
        }
        lines = [
            (f'{" " * 6}{key}={default},', f'  # <{dtype}> {dhelp}')
            for key, (dtype, default, dhelp) in members.items()
        ]
        maxlen = max(len(line[0]) for line in lines)
        msg = '\n'.join(sorted([
            f'{head:{maxlen}s}{tail}' for head, tail in lines
        ]))

        return f'\'\n{msg}\n\''

    @classmethod
    @property
    def defaults(cls):
        for key, dtype in cls._types.items():
            val = cls._defaults.get(key, None)
            yield (key, dtype(val) if val is not None else val)


OPTIMIZER_SETUPS = {
    'oneshot': (OneShotOptimizer, ()),
    'gd': (GradientDescentOptimizer, ('lr', 'n_iter')),
    'lbfgs': (LBFGSOptimizer, ('max_iter', 'max_eval', 'max_ls', 'gtol')),
    'tlbfgs': (TorchLBFGSOptimizer, ('lr', 'max_iter')),
    'mexicore': (EILVSOptimizer, ('gridsize', 'weighted', 'stabilize_interval', 'seq_reg', 'seq_reg_init')),
    'smo': (SMOOptimizer, ('stabilize_interval',)),
    'emicore': (EMICOREOptimizer, (
        'stabilize_interval',
        'gridsize',
        'pairsize',
        'samplesize',
        'core_trials',
        'corethresh',
        'corethresh_width',
        'smo_steps',
        'smo_axis',
    )),
}


ACQUISITION_FNS = {
    'lcb': (LowerConfidenceBound, {'beta': 0.1}),
    'alcb': (AdaptiveLCB, {'d': None}),
    'ei': (ExpectedImprovement, {}),
    'wei': (WeightedExpectedImprovement, {}),
}


class QCParams(OptionParams):
    n_layers: int = 1, 'Number of circuit layers'
    n_qbits: int = 2, 'Number of QBits'
    sector: int = -1, 'Sector -1 or 1'
    n_readout: int = 0, 'Number of shots'
    j_coupling: csobj(float, length=3) = '1,1,1', 'Nearest Neigh. interaction coupling'
    h_coupling: csobj(float, length=3) = '1,1,1', 'External magnetic field coupling'
    pbc: click.BOOL = True, 'Set Periodic/Open Boundary Conditions PBC or OBC. PBC default'
    circuit: click.Choice(['generic', 'esu2']) = 'generic', 'Circuit name'  # noqa: F821
    backend: click.Choice(list(BACKENDS)) = 'quest', 'Backend for QC'
    noise_level: float = 0.0, 'Circuit noise level'
    free_angles: int = None, 'number of free angles'
    assume_exact: click.BOOL = False, 'Assume energy is exact or an estimate.'
    cache: click.Path(dir_okay=False) = None, 'Cache for ground state wave function and initial train data.'
    train_data_mode: click.Choice(('cache', 'compute')) = 'compute', 'Inital data mode'  # noqa: F821


class KernelParams(OptionParams):
    sigma_0: PositiveFloat = None, 'Prior variance'
    gamma: PositiveFloat = 1.0, 'Kernel width parameter'


class GPParams(OptionParams):
    kernel: click.Choice(list(KERNELS)) = 'vqe', 'Name of the kernel'
    reg_term: float = 1e-10, 'Observation noise'
    reg_term_estimates: int = None, 'Number of estimates for reg_term'
    kernel_params: KernelParams() = '', 'Kernel options'
    prior_mean: click.BOOL = False, 'Setting non zero mean if True'


class AcqParams(OptionParams):
    func: click.Choice(['lcb', 'ei', 'alcb', 'wei']) = 'lcb', ''  # noqa: F821
    optim: click.Choice(list(OPTIMIZER_SETUPS)) = 'oneshot', ''
    lr: float = 1., ''
    n_iter: int = None, ''
    max_iter: int = 200, ''
    max_eval: int = None, ''
    max_ls: int = None, ''
    gtol: float = None, ''
    gridsize: int = None, ''
    weighted: click.BOOL = None, ''
    stabilize_interval: int = None, ''
    seq_reg: float = 0.0, ''
    seq_reg_init: int = -20, ''
    pairsize: int = 20, ''
    gridsize: int = 100, ''
    samplesize: int = 100, ''
    corethresh: float = 1.0, ''
    corethresh_width: int = 10, ''
    core_trials: int = 10, ''
    smo_steps: int = 100, ''
    smo_axis: click.BOOL = False, ''


class HyperParams(OptionParams):
    optim: click.Choice(['adam', 'grid', 'none']) = 'grid', ''  # noqa: F821
    loss: click.Choice(['mll', 'loo']) = 'loo', ''  # noqa: F821
    lr: float = 1e-4, ''
    threshold: float = 0.0, ''
    steps: int = 200, ''
    interval: str = '', ''
    max_gamma: float = 10.0, ''


class BOParams(OptionParams):
    train_samples: int = 5000, ''
    candidate_samples: int = 500, ''
    candidate_shots: int = 10, ''
    n_iter: int = 50, 'Iteration for Bayesian Optimization'
    acq_params: AcqParams() = '', ''
    hyperopt: HyperParams() = '', ''
    stabilize_interval: int = 0, 'Iteration for Optimization'
    iter_mode: click.Choice(['step', 'qc']) = 'step', ''  # noqa: F821
