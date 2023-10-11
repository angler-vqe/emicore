import logging
import json
import time
import math
from argparse import Namespace
from random import shuffle
from itertools import product, islice

import click
import torch
import h5py
import numpy as np
from tqdm import tqdm

from emicore.util import DataSampler
from emicore.cli import QCParams


def nakanishi_step(x_start, y_start, k_dim, true_energy):
    shift = torch.sparse_coo_tensor(list(zip(k_dim)), math.pi / 3., x_start.shape)
    x_shift = torch.cat([
        x_start - shift,
        x_start + shift,
    ])
    y_shift = true_energy(x_shift)

    spt = 3 ** .5 / 3
    x_pinv = torch.tensor([
        [-1.0, 1.0, 1.0],
        [2.0, -1.0, -1.0],
        [0.0, -spt, spt],
    ], dtype=torch.float64)
    y_train = torch.tensor([y_start, *y_shift])
    c0, c1, c2 = x_pinv @ y_train
    theta = torch.atan2(c2, c1) + math.pi

    shift_theta = torch.sparse_coo_tensor(list(zip(k_dim)), theta, x_start.shape)
    x_theta = (x_start + shift_theta) % math.tau
    y_theta = c0 + c1 * theta.cos() + c2 * theta.sin()

    return x_theta, y_theta


def dim_iterator(shape, random=False):
    indices = list(product(*(range(dim) for dim in shape)))
    while True:
        if random:
            shuffle(indices)
        for index in indices:
            yield index


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


@main.command('optimize')
@click.argument('output_file', type=click.Path(writable=True))
@QCParams.options()
@click.option('--n-iter', type=int, default=10, help='Iteration for Optimization')
@click.option('--stabilize-interval', type=int, default=0, help='Iteration for Optimization')
@click.option('--random/--sequential', default=False, help='Dimension selection strategy.')
@click.pass_context
def train(ctx, **kwargs):
    args = Namespace(**kwargs)

    start_time = time.time()

    sampler = DataSampler(
        args.n_qbits,
        args.n_layers,
        args.j_coupling,
        args.h_coupling,
        n_readout=args.n_readout,
        sector=args.sector,
        noise_level=args.noise_level,
        rng=ctx.obj.rng,
        circuit=args.circuit,
        pbc=args.pbc,
        cache_fname=args.cache
    )

    if args.train_data is not None:
        with h5py.File(args.train_data, 'r') as fd:
            x_start = torch.from_numpy(fd['x_train'][()])
            y_start = torch.from_numpy(fd['y_train'][()])
    else:
        x_start, y_start = sampler.sample(1)
    y_best = y_start

    # computes true wf and true energy for ground and first excited states
    true_e0, true_e1, true_wf = sampler.exact_diag()

    def observe_fn(x_start, y_start, step, exact=False):
        nonlocal y_best

        fidelity = sampler.exact_overlap(x_start)

        if exact:
            y_true = y_start.item()
        else:
            y_true = sampler.exact_energy(x_start).item()

        if y_start < y_best:
            y_best = y_start

        observables = {
            'n_qc_eval': 1 + (step + 1) * 2,
            'y_best': y_best.item(),
            'y_start': y_start.item(),
            'y_true': y_true,
            'fidelity': fidelity,
        }

        if args.stabilize_interval:
            observables['n_qc_eval'] += step // args.stabilize_interval

        if ctx.obj.json_log is not None:
            with open(ctx.obj.json_log, 'a') as fd:
                json.dump(observables, fd)
                fd.write('\n')

        step_params.append(x_start)
        pred_energy.append(y_start.item())
        true_energy.append(y_true)
        logging.info(f'Step {step:04d}, pred energy: {y_start.item():.2e}, true energy: {y_true:.2e}')

    pred_energy = []
    true_energy = []
    step_params = []
    k_dims = []
    for step, k_dim in tqdm(enumerate(islice(dim_iterator(x_start.shape, args.random), args.n_iter))):
        k_dims.append(int(np.ravel_multi_index(k_dim, x_start.shape)))
        do_reset = step and args.stabilize_interval and not (step % args.stabilize_interval)
        x_start, y_start = nakanishi_step(x_start, y_start, k_dim, sampler.true_energy)
        if do_reset:
            y_start = sampler.true_energy(x_start)
        observe_fn(
            x_start,
            y_start,
            step=step,
            exact=(do_reset and args.n_readout == 0 and args.noise_level == 0.0) or args.assume_exact
        )

    step_params = np.concatenate(step_params, axis=0)
    overlap = sampler.exact_overlap(step_params)

    logging.info('Nakanishi optimization ended successfully!')
    runtime = time.time() - start_time
    data_dict = {
        'x_train': x_start,
        'angles': step_params,
        'energy': pred_energy,
        'true_energy': true_energy,
        'true_e0': true_e0,
        'true_e1': true_e1,
        'overlap': overlap,
        'runtime': runtime,
        'n_qc_eval': 1 + (np.arange(len(overlap)) + 1) * 2,
        'k_last': k_dims
    }
    if args.stabilize_interval:
        data_dict['n_qc_eval'] += np.arange(len(overlap)) // args.stabilize_interval
    with h5py.File(args.output_file, 'w') as fd:
        for key, val in data_dict.items():
            fd[f'data/{key}'] = val
        fd['params'] = json.dumps(ctx.params)
    logging.info(f'Saved results to \'{args.output_file}\'.')


if __name__ == '__main__':
    main()
