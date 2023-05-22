import numpy as np
from scipy.linalg import eigh


class Gates:
    def __init__(self, n_qbits):
        super().__init__()
        self.n_qbits = n_qbits

        # helper variables
        self.dim = 2**n_qbits
        self.id = np.array([[1, 0], [0, 1]])
        self.sigmas = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])
        ]

    def ZERO(self):
        return np.zeros((self.dim, self.dim), dtype=np.complex64)

    def ID(self):
        return np.identity(self.dim, dtype=np.complex64)

    def _tensor_sigma(self, qbit, i):
        assert i <= 2, "no such pauli matrix exist"

        if qbit == 0:
            s = self.sigmas[i]
        else:
            s = self.id

        for j in range(1, self.n_qbits):
            if j == qbit:
                s = np.kron(self.sigmas[i], s)
            else:
                s = np.kron(self.id, s)

        return s

    def sX(self, qbit):
        return self._tensor_sigma(qbit, 0)

    def sY(self, qbit):
        return self._tensor_sigma(qbit, 1)

    def sZ(self, qbit):
        return self._tensor_sigma(qbit, 2)

    def RX(self, angle, qbit):
        return np.cos(angle / 2) * self.ID() - 1j * np.sin(angle / 2) * self.sX(qbit)

    def RY(self, angle, qbit):
        return np.cos(angle / 2) * self.ID() - 1j * np.sin(angle / 2) * self.sY(qbit)

    def RZ(self, angle, qbit):
        return np.cos(angle / 2) * self.ID() - 1j * np.sin(angle / 2) * self.sZ(qbit)

    def CNOT(self, qbit1, qbit2):
        a = 0.5 * (self.ID() + self.sZ(qbit1))
        b = 0.5 * np.dot(self.ID() - self.sZ(qbit1), self.sX(qbit2))
        return a + b

    def XX(self, qbit1, qbit2):
        return np.dot(self.sX(qbit1), self.sX(qbit2))

    def YY(self, qbit1, qbit2):
        return np.dot(self.sY(qbit1), self.sY(qbit2))

    def ZZ(self, qbit1, qbit2):
        return np.dot(self.sZ(qbit1), self.sZ(qbit2))

    def PXX(self, beta, qbit1, qbit2):
        return np.cos(beta / 2) * self.ID() - 1j * np.sin(beta / 2) * self.XX(qbit1, qbit2)

    def PYY(self, beta, qbit1, qbit2):
        return np.cos(beta / 2) * self.ID() - 1j * np.sin(beta / 2) * self.YY(qbit1, qbit2)

    def PZZ(self, beta, qbit1, qbit2):
        return np.cos(beta / 2) * self.ID() - 1j * np.sin(beta / 2) * self.ZZ(qbit1, qbit2)


class Circuit:
    def __init__(self):
        super().__init__()

    def __call__(self, *inputs):
        return self.run(*inputs)

    def run(self, *inputs):
        pass


class EmptyHeisenberg(Circuit):
    def __init__(self, n_qbits):
        super().__init__()
        self.n_qbits = n_qbits
        self.gates = Gates(n_qbits)

    def H(self, j, h, pbc=True):
        ham = self.gates.ZERO()

        if pbc:
            n_qbits = self.n_qbits
        else:
            n_qbits = self.n_qbits - 1

        for q in range(self.n_qbits):
            if h[0] != 0.:
                ham += h[0] * self.gates.sX(q)
            if h[1] != 0.:
                ham += h[1] * self.gates.sY(q)
            if h[2] != 0.:
                ham += h[2] * self.gates.sZ(q)

        for q in range(n_qbits):
            q1 = (q + 1) % self.n_qbits

            if j[0] != 0.:
                ham += j[0] * np.dot(self.gates.sX(q), self.gates.sX(q1))
            if j[1] != 0.:
                ham += j[1] * np.dot(self.gates.sY(q), self.gates.sY(q1))
            if j[2] != 0.:
                ham += j[2] * np.dot(self.gates.sZ(q), self.gates.sZ(q1))

        return ham


class HeisenbergCircuit(EmptyHeisenberg):
    def __init__(self, n_qbits, n_layers, angles, betas):
        super().__init__(n_qbits)
        self.n_layers = n_layers
        self.angles = angles
        self.betas = betas

    def run(self, wf):
        wf = wf.reshape(-1)
        wf = self.R0(self.angles[0], wf)

        for idx in range(self.n_layers):
            wf = self.RjXX(self.betas[idx], wf)
            wf = self.Rj(self.angles[idx + 1], wf)

        return wf

    def R0(self, angles, wf):
        for q in range(self.n_qbits):
            wf = np.dot(self.gates.RX(angles[0], q), wf)
            wf = np.dot(self.gates.RZ(angles[1], q), wf)
        return wf

    def Rj(self, angles, wf):
        for q in range(self.n_qbits):
            wf = np.dot(self.gates.RX(angles[0], q), wf)
            wf = np.dot(self.gates.RZ(angles[1], q), wf)
            wf = np.dot(self.gates.RY(angles[2], q), wf)
        return wf

    def RjXX(self, beta, wf):
        for q in range(self.n_qbits):
            q1 = (q + 1) % self.n_qbits
            wf = np.dot(self.gates.PXX(beta, q, q1), wf)
        return wf


def _initial_wf(n_qbits, momentum_sector=-1):
    dim = 2**n_qbits

    if momentum_sector == 1:
        wf = np.zeros(dim, dtype=np.complex64)
        wf[0] = 1.0
    elif momentum_sector == -1:
        zero = np.zeros(dim, dtype=np.complex64)

        if n_qbits % 2 == 0:
            alpha = (dim - 1) // 3
        else:
            alpha = (dim - 2) // 3

        zero[alpha] = -1
        zero[-alpha - 1] = 1

        wf = 1 / np.sqrt(2.0) * zero

    return wf


def measure_energy(n_qbits, n_layers, j, h, vector, mom_sector=-1, noise_level=0.0):
    if len(vector.shape) == 1:
        vector = vector[None]
    elif len(vector.shape) != 2:
        raise TypeError('Parameter values have to have 1 or 2 dimensions.')

    if noise_level != 0.0:
        raise ValueError("Python backend does not support noise.")

    energy = np.empty(vector.shape[0])
    for i in range(vector.shape[0]):
        energy[i], _, _ = measure_energy_wf_ham(n_qbits, n_layers, j, h, vector[i], mom_sector)

    return energy


def measure_energy_wf_ham(n_qbits, n_layers, j, h, vector, mom_sector=-1, pbc=True):
    # extract parameters
    angles = [vector[:2]]
    angles += [vector[ln * 4 + 3:ln * 4 + 6] for ln in range(n_layers)]
    betas = [vector[ln * 4 + 2] for ln in range(n_layers)]

    # initalize wf
    wf = _initial_wf(n_qbits, mom_sector)

    # build circuit
    circuit = HeisenbergCircuit(n_qbits, n_layers, angles, betas)

    # run circuit
    wf = circuit(wf)

    # measure energy
    energy = np.real(np.dot(np.conj(np.transpose(wf)), np.dot(circuit.H(j, h, pbc), wf)))

    return energy, wf, circuit.H(j, h, pbc)


def exact_spectrum(n_qbits, j, h, pbc, n_eigvals=2):
    circuit = EmptyHeisenberg(n_qbits)
    H = circuit.H(j, h, pbc)

    w, v = eigh(H, eigvals=(0, n_eigvals - 1))

    return w, v
