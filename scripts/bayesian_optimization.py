import logging
import json
import time
import signal
from argparse import Namespace
from functools import wraps


import click
import torch
import h5py
import numpy as np
from tqdm import tqdm

from emicore.bo import BayesianOptimization
from emicore.gp import GaussianProcess, KERNELS

from emicore.util import DataSampler, grid_search_gamma, interval_schedule
from emicore.cli import QCParams, GPParams, BOParams, ACQUISITION_FNS, OPTIMIZER_SETUPS, TrueSolution, Data
from emicore.cli import namedtuple_as_dict, final_property


@click.group()
@click.option('--seed', type=int, default=0xDEADBEEF)
@click.option('--json-log', type=click.Path(writable=True, dir_okay=False))
@click.pass_context
def main(ctx, seed, json_log):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('qiskit').setLevel(logging.WARNING)
    torch.manual_seed(seed)

    ctx.ensure_object(Namespace)
    ctx.obj.rng = np.random.default_rng(seed)

    if json_log is not None:
        ctx.obj.json_log = json_log
    else:
        ctx.obj.json_log = None

    def handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, handler)


def wrap_logger(log_fn):
    def wrapper(func):
        @wraps(func)
        def wrapped(model, state):
            error = func(model, state)
            log_fn(model, state, error)
            return error
        return wrapped
    return wrapper


class BayesOptCLI:
    def __init__(self, args, ctx):
        self.args = args
        self.ctx = ctx
        self._dict = {}

    @final_property
    def sampler(self):
        return DataSampler(
            self.args.n_qbits,
            self.args.n_layers,
            self.args.j_coupling,
            self.args.h_coupling,
            n_readout=self.args.n_readout,
            n_free_angles=self.args.free_angles,
            sector=self.args.sector,
            circuit=self.args.circuit,
            noise_level=self.args.noise_level,
            pbc=self.args.pbc,
            rng=self.ctx.obj.rng,
            cache_fname=self.args.cache
        )

    @final_property
    def acq_optimizer(self):
        params = self.args.acq_params

        acq_func_name = click.Choice(list(ACQUISITION_FNS))(getattr(params, 'func', 'lcb'))
        acq_func, acq_kwargs = ACQUISITION_FNS[acq_func_name]

        acquisition_optimizer_name = click.Choice(list(OPTIMIZER_SETUPS))(getattr(params, 'optim', 'oneshot'))
        optimizer_fn, optimizer_params = OPTIMIZER_SETUPS[acquisition_optimizer_name]
        optimizer_kwargs = {key: getattr(params, key) for key in optimizer_params}

        def candidate_sampler(score_fn=None):
            if score_fn is not None:
                raw_candidates = self.sampler.sample(
                    self.args.candidate_samples * self.args.candidate_shots, known=False
                )
                indices = torch.argsort(score_fn(raw_candidates), descending=True)[:self.args.candidate_samples]
                return raw_candidates[indices]
            return self.sampler.sample(self.args.candidate_samples, known=False)

        optimizer_kwargs.update({
            'acquisition_fn': acq_func(**acq_kwargs),
            'sampler': candidate_sampler
        })
        return optimizer_fn(**optimizer_kwargs)

    @final_property
    def kernel(self):
        self.args.kernel_params = Namespace(**self.args.kernel_params._asdict())
        if self.args.kernel_params.sigma_0 is None:
            self.args.kernel_params.sigma_0 = round(abs(self.sampler.estimate_ground_state_energy()))
        gamma = self.args.kernel_params.gamma
        sigma_0 = self.args.kernel_params.sigma_0

        kernel_kwargs = {
            'vqe': {'sigma_0': sigma_0, 'gamma': gamma},
            'sharedvqe': {'sigma_0': sigma_0, 'gamma': gamma, 'eta': 1.0, 'share': self.args.n_qbits},
            'rbf': {'sigma_0': sigma_0, 'gamma': gamma},
            'periodic': {'sigma_0': sigma_0, 'gamma': gamma},
            'matern': {'nu': 2.5},
            'torusrbf': {'sigma_0': sigma_0, 'gamma': gamma},
            'torusmatern': {'nu': 2.5},
        }[self.args.kernel]

        return KERNELS[self.args.kernel](**kernel_kwargs)

    @final_property
    def train_data(self):
        return Data(*self.sampler.cached_sample(
            self.args.train_samples, key='train', force_compute=self.args.train_data_mode == 'compute'
        ))

    @final_property
    def true_solution(self):
        return TrueSolution(*self.sampler.exact_diag())

    @final_property
    def model(self):
        prior_mean = self.sampler.estimate_ground_state_energy() if self.args.prior_mean else 0.0

        if self.args.reg_term_estimates is not None:
            logging.info('Estimating reg_term...')
            self.args.reg_term = self.sampler.estimate_variance(self.args.reg_term_estimates)
            logging.info(f'Estimated {self.args.reg_term:0.2e} for reg_term')

        model = GaussianProcess(
            self.train_data.x, self.train_data.y, kernel=self.kernel, reg=self.args.reg_term, mean=prior_mean
        )

        return model

    @final_property
    def log(self):
        log_keys = ('true_energy', 'pred_energy', 'best_params', 'gamma_history', 'n_qc_eval', 'k_last')
        return {key: [] for key in log_keys}

    @final_property
    def bayes_opt(self):

        def observe_fn(model, state, error):
            fidelity = self.sampler.exact_overlap(state['x_start'][None].numpy()).item()
            if self.args.assume_exact:
                y_true = state.get('y_start', 0.).item()
            else:
                y_true = self.sampler.exact_energy(state['x_start'][None].numpy()).item()
            observables = {
                'n_qc_eval': len(model.x_train),
                'y_best': state.get('y_best', 0.).item(),
                'y_start': state.get('y_start', 0.).item(),
                'y_true': y_true,
                'fidelity': fidelity,
            }
            for key, value in model.kernel.param_dict().items():
                observables[f'kernel.{key}'] = value

            simple_keys = {
                'acq_min',
                'min_ran',
                'n_added',
                'y_improve',
            }
            for key in simple_keys.intersection(state):
                observables[key] = state[key]

            if 'acq_max' in state and 'y_improve' in state:
                observables['acq_mismatch'] = state['acq_max'] - state['y_improve']
            if 'x_start' in state and 'k_best' in state:
                observables['k_last'] = int(np.ravel_multi_index(state['k_best'], state['x_start'].shape))

            if self.ctx.obj.json_log is not None:
                with open(self.ctx.obj.json_log, 'a') as fd:
                    json.dump(observables, fd)
                    fd.write('\n')

            logging.warning(
                f'Step {state["step"]:04d}, '
                f'last energy: {observables["y_start"]:.3e}, '
                f'true energy: {y_true:.3e}, '
                f'best energy: {observables["y_best"]:.3e}, '
                f'gamma: {observables.get("kernel.gamma")}'
            )

            storables = (
                ('true_energy', y_true),
                ('pred_energy', observables['y_start']),
                ('best_params', state['x_start']),
                ('gamma_history', observables.get('kernel.gamma')),
                ('n_qc_eval', len(model.x_train)),
                ('k_last', observables.get('k_last')),
            )
            for key, value in storables:
                self.log[key].append(value)

        @wrap_logger(observe_fn)
        def rms_error_fn(model, state):
            return 0.0

        if self.args.reg_term == 0.0 and self.args.acq_params.stabilize_interval:
            raise RuntimeError(
                'Exact GPs cannot be used with a stabilization interval! The resulting Gram matrix will become non-PD!'
            )

        return BayesianOptimization(
            model=self.model,
            optimizer=self.acq_optimizer,
            true_fn=self.sampler.true_energy,
            error_fn=rms_error_fn
        )

    @final_property
    def interval(self):
        return interval_schedule(self.args.hyperopt.interval)

    @final_property
    def lossfn(self):
        return {
            'loo': self.model.loocv_mll_closed,
            'mll': self.model.log_likelihood,
        }[self.args.hyperopt.loss]

    @final_property
    def kernel_optim(self):
        if self.args.hyperopt.optim != 'adam':
            raise RuntimeError(f'Cannot use kernel_optim with {self.args.hyperopt.optim}!')
        params = list(self.kernel.parameters())
        for param in params:
            param.requires_grad = True
        if not params:
            logging.warning(f'{self.kernel} has no hyper parameters to optimize!')
        return torch.optim.Adam(params, lr=self.args.hyperopt.lr)

    @final_property
    def n_iter(self):
        if self.args.iter_mode == 'qc' and self.args.acq_params.optim == 'mexicore':
            # TODO: this does not consider stabilize_interval
            return (self.args.n_iter - self.args.train_samples) // 2

        return self.args.n_iter

    def hyperopt(self, step):
        if self.args.hyperopt.optim == 'grid':
            if self.interval(step):
                max_gamma = self.args.hyperopt.max_gamma
                wiggle = np.random.normal(0, (max_gamma - 1) / self.args.hyperopt.steps)
                grid_search_gamma(
                    self.model,
                    min_gamma=1.0,
                    max_gamma=max_gamma + wiggle,
                    num=self.args.hyperopt.steps,
                    loss=self.args.hyperopt.loss
                )
        elif self.args.hyperopt.optim == 'adam':
            if self.interval(step):
                for step in range(self.args.hyperopt.steps):
                    self.kernel_optim.zero_grad()
                    loss = self.lossfn()
                    loss.backward()
                    self.kernel_optim.step()
                    self.model.reinit()


@main.command('train')
@click.argument('output_file', type=click.Path(writable=True))
@QCParams.options()
@GPParams.options()
@BOParams.options()
@click.pass_context
def train(ctx, **kwargs):
    args = Namespace(**kwargs)
    ns = BayesOptCLI(args, ctx)
    start_time = time.time()

    ns.log['gamma_history'].append(ns.model.kernel.gamma.detach().item())

    try:
        for bayes_step in tqdm(range(ns.n_iter)):
            ns.hyperopt(bayes_step)
            ns.bayes_opt.step(bayes_step)
    finally:

        best_params = torch.stack(ns.log['best_params'], dim=0)
        overlap = ns.sampler.exact_overlap(best_params.numpy())

        logging.info('Bayesian Optimization ended successfully')
        runtime = time.time() - start_time

        state_dict = ns.model.state_dict()
        data_dict = {
            'x_train': ns.model.x_train,
            'y_train': ns.model.y_train,
            'angles': best_params,
            'energy': ns.log['pred_energy'],
            'true_energy': ns.log['true_energy'],
            'true_e0': ns.true_solution.e0,
            'true_e1': ns.true_solution.e1,
            'overlap': overlap,
            'runtime': runtime,
            'gamma': ns.log['gamma_history'][1:],
            'n_qc_eval': ns.log['n_qc_eval'],
        }

        if args.reg_term_estimates is not None and args.reg_term_estimates > 0:
            ctx.params['reg_term'] = ns.model.reg
        if args.acq_params.optim == 'mexicore':
            data_dict['k_last'] = ns.log['k_last']

        with h5py.File(args.output_file, 'w') as fd:
            for key, val in state_dict.items():
                fd[f'state/{key}'] = val
            for key, val in data_dict.items():
                fd[f'data/{key}'] = val
            fd['params'] = json.dumps(namedtuple_as_dict(ctx.params))
        logging.info(f'Saved results to \'{args.output_file}\'.')


@main.command('init-cache')
@QCParams.options()
@GPParams.options()
@BOParams.options()
@click.pass_context
def init_cache(ctx, **kwargs):
    args = Namespace(**kwargs)
    args.train_data_mode = 'cache'
    ns = BayesOptCLI(args, ctx)

    if args.cache is None:
        logging.info('Nothing initialized!')
    elif ns.train_data is not None and ns.true_solution is not None:
        logging.info(f'Initialized {args.cache}.')


if __name__ == '__main__':
    main()
