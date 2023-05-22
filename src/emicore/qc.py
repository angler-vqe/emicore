from itertools import product
from functools import reduce
from operator import add, xor
from collections import deque

import numpy as np
from scipy.linalg import eigh

from qiskit import QuantumCircuit, QuantumRegister, Aer, IBMQ, execute
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.ibmqbackend import IBMQSimulator

from qiskit.opflow import X, Y, Z, I, CircuitStateFn, StateFn, PauliExpectation, AerPauliExpectation, CircuitSampler
from qiskit.quantum_info import state_fidelity
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter


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


def heisenberg_hamiltonian(n_qbits, j=[1.0, 1.0, 1.0], h=[0.0, 0.0, 1.0], pbc=True):
    """Constructs the Heisenberg Hamiltonian with arbitrary external fields

    Parameters
    ----------
    j: list of float
        The couplings for the nearest neighbor terms XX, YY, ZZ
    h: list of float
        The parameters for the field in X, Y and Z direction
    pbc: bool
        Flag indicating whether periodic boundary conditions should be used or not

    Returns
    -------
    PauliOp: obj:`qiskit.opflow.PauliOp`
        The hamiltonian.
    """
    summands = []
    for op, s_coeff, p_coeff in zip((X, Y, Z), h, j):
        single = deque((op,) + (I,) * (n_qbits - 1))
        paired = deque((op, op) + (I,) * (n_qbits - 2))
        for _ in range(n_qbits):
            summands.append(s_coeff * reduce(xor, single))
            summands.append(p_coeff * reduce(xor, paired))
            single.rotate()
            paired.rotate()
        if not pbc:
            del summands[-1]

    return reduce(add, summands)


def generic_circuit(n_qbits, n_layers):
    """Construct a generic quantum circuit

    Parameters
    ----------
    n_qbits: Int
        Total number of qbits in the system
    n_layers: Int
        Total number of (entangling and rotation) layers in the ansatz
    """
    reg = QuantumRegister(n_qbits)
    circuit = QuantumCircuit(reg)
    params = [[Parameter(f'θ[{gate}, {qbit}]') for qbit in range(n_qbits)] for gate in range(2 + n_layers * 4)]

    for i in range(n_qbits):
        circuit.rx(params[0][i], reg[i])
        circuit.rz(params[1][i], reg[i])
    circuit.barrier()
    for layer in range(n_layers):
        offset = 2 + layer * 4
        for i in range(n_qbits):
            circuit.rxx(params[offset][i], reg[i], reg[(i + 1) % n_qbits])
        circuit.barrier()
        for i in range(n_qbits):
            circuit.rx(params[offset + 1][i], reg[i])
            circuit.rz(params[offset + 2][i], reg[i])
            circuit.ry(params[offset + 3][i], reg[i])
    return circuit


def make_circuit(n_qbits, n_layers, vector, mom_sector=1, circuit='generic'):
    reg = QuantumRegister(n_qbits, 'q')
    ansatz = QuantumCircuit(reg)
    if mom_sector == -1:
        # ansatz.pauli('X' * (n_qbits // 2), [register[i] for i in range(1, n_qbits, 2)])
        ansatz.pauli(('IX' * ((n_qbits + 1) // 2))[:n_qbits], reg)
    elif mom_sector != 1:
        raise RuntimeError(f'Unsupported sector: \'{mom_sector}\'')

    param_keys = []
    if circuit == 'generic':
        qcircuit = generic_circuit(n_qbits, n_layers)
        param_keys += [f'θ[{gate}, {qbit}]' for gate in range(2 + n_layers * 4) for qbit in range(n_qbits)]
    elif circuit == 'esu2':
        qcircuit = EfficientSU2(
            num_qubits=n_qbits,
            su2_gates=None,
            entanglement='full',
            insert_barriers=True,
            reps=n_layers,
            parameter_prefix='θ'
        )
        param_keys += [f'θ[{n}]' for n in range((2 + n_layers * 2) * n_qbits)]
    else:
        raise RuntimeError(f'Qiskit only supports circuits \'generic\' and \'esu2\', got \'{circuit}\'')

    param_reference = {parameter.name: parameter for parameter in qcircuit.parameters}
    param_dict = dict(zip([param_reference[key] for key in param_keys], vector.flatten()))
    qcircuit = qcircuit.assign_parameters(param_dict)
    ansatz = ansatz.compose(qcircuit)
    return ansatz


def compute_energy(hamiltonian, circuit, n_readout=1024, noisy=False):
    """Given a Hamiltonian and a quantum circuit encoding the wave function, compute the energy

    Parameters
    ----------
    hamiltonian: WeightedPauliOperator
        The Hamiltonian of the system (has to be nonempty)
    circuit: QuantumCircuit
        Quantum circuit encoding the wave function
    n_readout: int
        Number of samples taken in the measurement
    """
    simkw = {}
    if noisy:
        if not IBMQ.providers():
            IBMQ.load_account()
        provider = IBMQ.get_provider()
        device_backend = next(
            elem
            for elem in provider.backend.backends()
            if not isinstance(elem, IBMQSimulator)
        )
        simkw['noise_model'] = NoiseModel.from_backend(device_backend, thermal_relaxation=False)
        simkw['coupling_map'] = device_backend.configuration().coupling_map
        simkw['basis_gates'] = simkw['noise_model'].basis_gates

    backend = QasmSimulator(**simkw)
    operator = StateFn(hamiltonian, is_measurement=True)
    psi = CircuitStateFn(circuit)

    if n_readout > 0:
        q_instance = QuantumInstance(backend, shots=n_readout)
        expectation = PauliExpectation().convert(operator @ psi)
        sampler = CircuitSampler(q_instance).convert(expectation)
    else:
        expectation = AerPauliExpectation().convert(operator @ psi)
        sampler = CircuitSampler(backend).convert(expectation)

    energy = sampler.eval().real

    return energy


def compute_overlap(exact_wf, circuit):
    """Computes the overlap between a state vector and the output state vector of a circuit

    Parameters
    ----------
    exact_wf: obj:`numpy.ndarray`
        State vector as a complex numpy array
    circuit: QuantumCircuit
        Quantum circuit encoding the wave function
    """
    backend = Aer.get_backend('statevector_simulator')
    wf = execute(circuit, backend=backend).result().get_statevector()
    return state_fidelity(exact_wf, wf)


def compute_state_vector(circuit):
    """Computes the state vector of a circuit

    Parameters
    ----------
    circuit: QuantumCircuit
        Quantum circuit encoding the wave function
    """
    backend = Aer.get_backend('statevector_simulator')
    wf = execute(circuit, backend=backend).result().get_statevector()
    return wf


def param_shift(
    hamiltonian,
    angles,
    n_qbits,
    n_layers,
    mom_sector=1,
    circuit='generic',
    n_readout=1024,
    noisy=False,
):
    def measure(angles):
        energies = []
        for batch in angles:
            qcircuit = make_circuit(n_qbits, n_layers, batch, mom_sector=mom_sector, circuit=circuit)
            energy = compute_energy(hamiltonian, qcircuit, n_readout=n_readout, noisy=noisy)
            energies.append(energy)
        return np.stack(energies, axis=0)

    grad = np.zeros_like(angles)
    for index in product((slice(None),), *(range(s) for s in angles.shape[1:])):
        org = angles[index].copy()
        angles[index] += np.pi / 2.
        grad[index] = measure(angles) / 2.
        angles[index] -= np.pi
        grad[index] -= measure(angles) / 2.
        angles[index] = org

    return grad


def measure_energy(
    n_qbits,
    n_layers,
    j,
    h,
    angles,
    mom_sector=1,
    noise_level=0.0,
    prob_0to1=0.0,
    prob_1to0=0.0,
    n_readout=0,
    pbc=True,
    circuit='generic'
):
    angles = expand_params(angles, n_qbits)

    energies = []
    for batch in angles:
        hamiltonian = heisenberg_hamiltonian(n_qbits, j, h, pbc=pbc)
        qcircuit = make_circuit(n_qbits, n_layers, batch, mom_sector=mom_sector, circuit=circuit)
        energy = compute_energy(
            hamiltonian, qcircuit, n_readout=n_readout, noisy=noise_level != 0.
        )
        energies.append(energy)
    return np.array(energies)


def measure_energy_variance(
    n_qbits,
    n_layers,
    j,
    h,
    angles,
    mom_sector=1,
    n_readout=0,
    pbc=True,
    circuit='generic'
):
    angles = expand_params(np.array(angles), n_qbits)

    if n_readout <= 0:
        energies = measure_energy(
            n_qbits,
            n_layers,
            j,
            h,
            angles,
            mom_sector=mom_sector,
            noise_level=0.0,
            prob_0to1=0.0,
            prob_1to0=0.0,
            n_readout=n_readout,
            pbc=pbc,
            circuit=circuit
        )
        return energies, np.zeros_like(energies)

    energies = []
    variances = []
    for batch in angles:
        hamiltonian = heisenberg_hamiltonian(n_qbits, j, h, pbc=pbc)
        qcircuit = make_circuit(n_qbits, n_layers, batch, mom_sector=mom_sector, circuit=circuit)

        operator = StateFn(hamiltonian, is_measurement=True)
        psi = CircuitStateFn(qcircuit)

        q_instance = QuantumInstance(QasmSimulator(), shots=n_readout)
        expectation = PauliExpectation().convert(operator @ psi)
        sampler = CircuitSampler(q_instance).convert(expectation)

        energies.append(sampler.eval().real)
        variances.append(PauliExpectation().compute_variance(sampler).real / n_readout)

    return np.array(energies), np.array(variances)


def parameter_shift_gradient(
    n_qbits,
    n_layers,
    j,
    h,
    angles,
    mom_sector=1,
    noise_level=0.0,
    prob_0to1=0.0,
    prob_1to0=0.0,
    n_readout=0,
    pbc=True,
    circuit='generic'
):
    angles = expand_params(angles, n_qbits)

    hamiltonian = heisenberg_hamiltonian(n_qbits, j, h)
    gradients = param_shift(
        hamiltonian,
        angles,
        n_qbits,
        n_layers,
        mom_sector=mom_sector,
        circuit=circuit,
        n_readout=n_readout,
        noisy=False
    )
    return gradients


def measure_overlap(
    n_qbits,
    n_layers,
    angles,
    exact_wf,
    mom_sector=1,
    circuit='generic'
):
    angles = expand_params(angles, n_qbits)

    overlaps = []
    for batch in angles:
        qcircuit = make_circuit(n_qbits, n_layers, batch, mom_sector=mom_sector, circuit=circuit)
        overlaps.append(compute_overlap(exact_wf, qcircuit))
    return np.array(overlaps)


def measure_state_vector(
    n_qbits,
    n_layers,
    angles,
    mom_sector=1,
    circuit='generic'
):
    angles = expand_params(angles, n_qbits)

    state_vectors = []
    for batch in angles:
        qcircuit = make_circuit(n_qbits, n_layers, batch, mom_sector=mom_sector, circuit=circuit)
        state_vector = compute_state_vector(qcircuit)
        state_vectors.append(state_vector)
    return np.stack(state_vectors, axis=0)


def exact_spectrum(n_qbits, j=[1.0, 1.0, 1.0], h=[0.0, 0.0, 1.0], pbc=True, n_eigvals=2):
    ham = heisenberg_hamiltonian(n_qbits, j, h, False).to_matrix()
    w, v = eigh(ham, eigvals=(0, n_eigvals - 1))
    return w, v
