from src.emicore.energy import BACKENDS
from src.emicore.utils import circuit_param_size


_actions = {}


def register_action(name):
    def wrapped(func):
        _actions[name] = func
        return func
    return wrapped


class Action:
    def evaluate(self, field):
        pass

    def __call__(self, field):
        return self.evaluate(field)


@register_action("heisenberg")
class Heisenberg(Action):
    def __init__(
        self,
        n_qbits,
        n_layers,
        j,
        h,
        momentum_sector,
        noise_level=0.0,
        n_reps=10,
        prob_1to0=0.0,
        prob_0to1=0.0,
        n_readout=0,
        translational_inv=True,
        pbc=True,
        circuit='generic',
        backend='quest'
    ):
        super().__init__()
        self.translational_inv = translational_inv
        self.n_params = circuit_param_size(circuit, n_layers)
        self.n_qbits = n_qbits
        self.energy_fn = BACKENDS[backend](
            n_qbits=n_qbits,
            n_layers=n_layers,
            j=j,
            h=h,
            circuit=circuit,
            mom_sector=momentum_sector,
            noise_level=noise_level,
            n_reps=n_reps,
            prob_1to0=prob_1to0,
            prob_0to1=prob_0to1,
            n_readout=n_readout,
            pbc=pbc
        )

    def evaluate(self, angles):
        if not self.translational_inv:
            try:
                angles = angles.reshape(angles.shape[0], self.n_params, self.n_qbits)
            except ValueError as err:
                raise ValueError("Number of parameters does not match the non translational invariant case") from err
        return self.energy_fn(angles)
