import math
from functools import partial

import torch
import numpy as np
from src.qc.numpy import exact_spectrum
from src.qc.qiskit import measure_energy_variance, measure_overlap, measure_energy
from src.energy import BACKENDS
from src.utils import circuit_param_size
from src.cli import arrcache


class DataSampler:
    def __init__(
        self,
        n_qbits,
        n_layers,
        j=(1., 1., 1.),
        h=(1., 1., 1.),
        n_free_angles=None,
        rng=None,
        sector=-1,
        noise_level=0.,
        prob_1to0=0.,
        prob_0to1=0.,
        n_readout=0,
        pbc=True,
        circuit='generic',
        backend='quest',
        cache_fname='',
    ):
        n_circuit_params = circuit_param_size(circuit, n_layers)
        if n_free_angles is None:
            n_free_angles = n_circuit_params * n_qbits
        self.n_free_angles = min(n_circuit_params * n_qbits, n_free_angles)
        if rng is None:
            rng = np.random.default_rng()
        self.x_template = rng.uniform(0, 2 * math.pi, (1, n_circuit_params, n_qbits))
        self.rng = rng

        self.kwargs = {
            'n_qbits': n_qbits,
            'n_layers': n_layers,
            'j': j,
            'h': h,
            'circuit': circuit,
            'mom_sector': sector,
            'noise_level': noise_level,
            'prob_1to0': prob_1to0,
            'prob_0to1': prob_0to1,
            'n_readout': n_readout,
            'pbc': pbc
        }
        self.backend = backend

        self.energy_fn = BACKENDS[backend](**self.kwargs)
        self.energy_var_fn = partial(
            measure_energy_variance,
            n_qbits,
            n_layers,
            j,
            h,
            mom_sector=self.kwargs['mom_sector'],
            n_readout=self.kwargs['n_readout'],
            pbc=self.kwargs['pbc'],
            circuit=self.kwargs['circuit'],
        )

        self.cache_fname = cache_fname
        self._exact_diag = None

    def true_energy(self, angles):
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        return self.energy_fn(angles)

    def true_energy_variance(self, angles):
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        return self.energy_var_fn(angles)

    def sample(self, n_samples=1000, known=True):
        # expand and fill in template
        x_data = self.x_template.repeat(n_samples, axis=0)
        x_data.reshape(
            x_data.shape[0], np.prod(x_data.shape[1:], dtype=int)
        )[:, :self.n_free_angles] = self.rng.uniform(0, 2 * math.pi, (n_samples, self.n_free_angles))

        retval = torch.from_numpy(x_data)
        # compute true values if known
        if known:
            retval = (retval, self.true_energy(x_data))
        return retval

    def sample_linspace(self, n_samples=50, axes=(0, 0), known=True):
        # expand and fill in template with linspace
        x_data = self.x_template.repeat(n_samples, axis=0)
        x_data[(slice(None),) + axes] = torch.linspace(0, 2 * math.pi, n_samples)

        retval = torch.from_numpy(x_data)
        # compute true values if known
        if known:
            retval = (retval, self.true_energy(x_data))
        return retval

    def estimate_variance(self, n_samples):
        x_reg_est = self.sample(n_samples, known=False)
        _, y_reg_est_var = self.true_energy_variance(x_reg_est)
        return y_reg_est_var.mean(0)

    def estimate_ground_state_energy(self):
        return -1.24 * self.kwargs['n_qbits'] - 0.29

    def exact_diag(self):
        if self._exact_diag is None:
            args = (
                int(self.kwargs['n_qbits']),
                tuple(self.kwargs['j']),
                tuple(self.kwargs['h']),
                bool(self.kwargs['pbc']),
            )
            self._exact_diag = arrcache(
                self.cache_fname,
                lambda: exact_diag(*args),
                args,
                keys=('true_e0', 'true_e1', 'true_wf')
            )
        return self._exact_diag

    def exact_overlap(self, angles):
        _, _, exact_wf = self.exact_diag()

        return measure_overlap(
            self.kwargs['n_qbits'],
            self.kwargs['n_layers'],
            angles=np.array(angles),
            exact_wf=exact_wf,
            mom_sector=self.kwargs['mom_sector'],
            circuit=self.kwargs['circuit'],
        )

    def exact_energy(self, angles):
        return measure_energy(
            self.kwargs['n_qbits'],
            self.kwargs['n_layers'],
            self.kwargs['j'],
            self.kwargs['h'],
            angles=np.array(angles),
            pbc=self.kwargs['pbc'],
            mom_sector=self.kwargs['mom_sector'],
            circuit=self.kwargs['circuit'],
        )

    @property
    def _cache_identifiers(self):
        return (
            int(self.kwargs['n_qbits']),
            int(self.kwargs['n_layers']),
            tuple(self.kwargs['j']),
            tuple(self.kwargs['h']),
            int(self.kwargs['n_readout']),
            int(self.n_free_angles),
            int(self.kwargs['mom_sector']),
            str(self.backend),
            str(self.kwargs['circuit']),
            float(self.kwargs['noise_level']),
            bool(self.kwargs['pbc']),
        )

    def cached_sample(self, n_samples, key='train', force_compute=False):
        return arrcache(
            self.cache_fname if not force_compute else None,
            lambda: self.sample(n_samples, known=True),
            (key, n_samples, *self._cache_identifiers),
            keys=(f'x_{key}', f'y_{key}')
        )


def plot_gp(ax, mean, covar, x_test, kernel, color='b'):
    yerr = 2 * np.diag(covar) ** .5
    ax.plot(x_test, mean, color=color, label=f"{kernel}")
    ax.fill_between(x_test, mean + yerr, mean - yerr, alpha=0.3, color=color)


def exact_diag(n_qbits, j, h, pbc=True):
    exact_eigvals, exact_eigvecs = exact_spectrum(n_qbits, j, h, pbc=pbc)
    true_e0 = exact_eigvals[0]
    true_e1 = exact_eigvals[1]
    true_wf = exact_eigvecs[:, 0]
    return true_e0, true_e1, true_wf
