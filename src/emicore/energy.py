import numpy as np
import torch

from src.emicore.qc import measure_energy as measure_energy_qiskit
from src.emicore.qc import parameter_shift_gradient as parameter_shift_gradient_qiskit

BACKENDS = {}


def register_backend(name):
    def wrapped(func):
        BACKENDS[name] = func
        return func
    return wrapped


class MeasureEnergyQiskit(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        angles,
        n_qbits,
        n_layers,
        j,
        h,
        mom_sector=-1,
        noise_level=0.0,
        n_reps=1,
        prob_1to0=0.0,
        prob_0to1=0.0,
        n_readout=0,
        pbc=True,
        circuit='generic'
    ):
        ctx.save_for_backward(angles)
        ctx.n_qbits = n_qbits
        ctx.n_layers = n_layers
        ctx.j = j
        ctx.h = h
        ctx.mom_sector = mom_sector
        ctx.noise_level = noise_level
        ctx.n_reps = n_reps
        ctx.prob_0to1 = prob_0to1
        ctx.prob_1to0 = prob_1to0
        ctx.n_readout = n_readout
        ctx.pbc = pbc
        ctx.circuit = circuit

        return torch.tensor(
            measure_energy_qiskit(
                n_qbits,
                n_layers,
                j,
                h,
                angles.detach().cpu().numpy(),
                mom_sector,
                noise_level,
                prob_0to1,
                prob_1to0,
                n_readout,
                pbc,
                circuit
            )
        ).to(angles)

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors

        shape = angles.shape

        # gradient is measured n_reps times to reduce variance
        angles = angles.repeat_interleave(ctx.n_reps, 0)
        grad_input = parameter_shift_gradient_qiskit(
            ctx.n_qbits,
            ctx.n_layers,
            ctx.j,
            ctx.h,
            angles.detach().cpu().numpy(),
            ctx.mom_sector,
            ctx.noise_level,
            ctx.prob_0to1,
            ctx.prob_1to0,
            ctx.n_readout,
            ctx.pbc,
            ctx.circuit
        )
        grad_input = torch.tensor(grad_input).to(angles)

        grad_input = grad_input.reshape(ctx.n_reps, *shape).mean(0)
        grad_output = grad_output[(...,) + (None,) * (len(grad_input.shape) - 1)]

        return (grad_output * grad_input,) + (None,) * 12


class EnergyFunction:
    '''Abstract class to define energy functions.'''
    function = None

    def __init__(
        self,
        n_qbits,
        n_layers,
        j,
        h,
        circuit='generic',
        mom_sector=-1,
        noise_level=0.0,
        n_reps=1,
        prob_1to0=0.0,
        prob_0to1=0.0,
        n_readout=0,
        pbc=True
    ):
        self._n_qbits = n_qbits
        self._n_layers = n_layers
        self._j = j
        self._h = h
        self._mom_sector = mom_sector
        self._noise_level = noise_level
        self._n_reps = n_reps
        self._prob_1to0 = prob_1to0
        self._prob_0to1 = prob_0to1
        self._n_readout = n_readout
        self._pbc = pbc
        self._circuit = circuit

    def __call__(self, angles):
        is_numpy = isinstance(angles, np.ndarray)
        if is_numpy:
            angles = torch.from_numpy(angles)
        result = self.function(
            angles,
            self._n_qbits,
            self._n_layers,
            self._j,
            self._h,
            self._mom_sector,
            self._noise_level,
            self._n_reps,
            self._prob_1to0,
            self._prob_0to1,
            self._n_readout,
            self._pbc,
            self._circuit
        )
        if is_numpy:
            result = result.detach().cpu().numpy()
        return result


@register_backend('qiskit')
class EnergyQiskit(EnergyFunction):
    function = MeasureEnergyQiskit.apply
