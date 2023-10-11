import logging
from argparse import Namespace

import click
import torch
import h5py
import numpy as np

from emicore.util import DataSampler
from emicore.cli import QCParams, BOParams, Data
from emicore.cli import final_property


@click.group()
@click.option('--seed', type=int, default=0xDEADBEEF)
@click.pass_context
def main(ctx, seed):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('qiskit').setLevel(logging.WARNING)
    torch.manual_seed(seed)

    ctx.ensure_object(Namespace)
    ctx.obj.rng = np.random.default_rng(seed)


class GenerateCLI:
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
            sector=self.args.sector,
            circuit=self.args.circuit,
            noise_level=self.args.noise_level,
            pbc=self.args.pbc,
            rng=self.ctx.obj.rng,
            cache_fname=self.args.cache
        )

    @final_property
    def train_data(self):
        return Data(*self.sampler.sample(self.args.train_samples))


@main.command('make-train')
@click.argument('output')
@QCParams.options()
@click.option('--train-samples', type=int, default=1, help='Number of training samples.')
@click.pass_context
def make_train(ctx, **kwargs):
    args = Namespace(**kwargs)
    ns = GenerateCLI(args, ctx)

    with h5py.File(args.output, 'w') as fd:
        fd['x_train'] = ns.train_data.x
        fd['y_train'] = ns.train_data.y

    logging.info(f'Initialized {args.output}.')


if __name__ == '__main__':
    main()
