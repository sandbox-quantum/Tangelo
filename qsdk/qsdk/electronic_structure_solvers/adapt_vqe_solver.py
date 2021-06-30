"""
ADAPT-VQE algorithm framework, to solve electronic structure calculations.
https://www.nature.com/articles/s41467-019-10988-2
"""

from scipy.optimize import minimize
import numpy as np
from openfermion import commutator

from qsdk.electronic_structure_solvers.vqe_solver import Ansatze, VQESolver
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import pauliword_to_circuit
from agnostic_simulator import Circuit
from qsdk.toolboxes.operators import QubitOperator
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz


def get_pool(qubit_hamiltonian, n_qubits):
    """Use Hamiltonian to identify non-commuting Pauli strings to use as operator pool.
    We identify terms with even number of Y-operators, in order to define excitations
    which preserve T-reversal symmetry. We remove all Z operators, and we flip the first
    X or Y to its partner (i.e. X <> Y).
    Args:
        qubit_hamiltonian (QubitOperator): input Hamiltonian
        n_qubits (int): number of qubits for Hamiltonian

    Returns:
        pool_generators (list of QubitOperator): list of generators
    """

    pauli_lookup = {'Z': 1, 'X': 2, 'Y': 3}
    pauli_reverse_lookup = ['I', 'Z', 'X', 'Y']

    pool_generators, pool_tuples = list(), list()
    indices = list()

    for term in qubit_hamiltonian.terms:
        pauli_string = np.zeros(n_qubits, dtype=int)

        # identify all X or Y factors
        for index, action in term:
            if pauli_lookup[action] > 1:
                pauli_string[index] = pauli_lookup[action]

        # only allow one operator acting on a given set of qubits
        action_mask = tuple(pauli_string > 1)
        if action_mask in indices:
            continue

        # only consider terms with even number of Y
        if sum(pauli_string) % 2 == 0 and sum(pauli_string) > 0:
            # identify qubit operator to change X<>Y
            flip_index = np.where(pauli_string > 1)[0][0]
            pauli_string[flip_index] += (-1) ** (pauli_string[flip_index] % 2)

            # update set of used qubit combinations
            indices.append(action_mask)
            # create QubitOperator for the new generator
            operator_tuple = tuple(
                [(index, pauli_reverse_lookup[pauli]) for index, pauli in enumerate(pauli_string) if pauli > 0])
            # We don't use the coefficients directly, so since we need to multiply by 1.j for evaluating gradients,
            # I'm just instantiating these with that coefficient in place
            pool_generators.append(QubitOperator(operator_tuple, 1.0j))
            pool_tuples.append(operator_tuple)

    return pool_generators, pool_tuples


def rank_pool(pool_commutators, circuit, backend, tolerance=1e-3):
    gradient = [abs(backend.get_expectation_value(element, circuit)) for element in pool_commutators]
    max_partial = max(gradient)
    print(f'LARGEST PARTIAL DERIVATIVE: {max_partial :4E} \t[{gradient.index(max_partial)}]')
    return gradient.index(max_partial) if max_partial >= tolerance else -1


class AdaptAnsatz(Ansatz):

    def __init__(self, n_spinorbitals, n_electrons, operators=list()):

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.operators = operators

        self.var_params = None
        self.circuit = None

    @property
    def n_var_params(self):
        return len(self.operators)

    def set_var_params(self, var_params=None):
        """ Set initial variational parameter values. Defaults to zeros. """
        if var_params is None:
            var_params = np.zeros(self.n_var_params, dtype=float)
        elif var_params.size != self.n_var_params:
            raise ValueError('Invalid number of parameters.')
        self.var_params = var_params
        return var_params

    def update_var_params(self, var_params):
        """ Update variational parameters (done repeatedly during VQE) """
        pass
        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]

    def prepare_reference_state(self):
        """ Prepare a circuit generating the HF reference state. """
        # TODO non hardcoded mapping
        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping='JW')

    def build_circuit(self, var_params=None):
        """ Construct the variational circuit to be used as our ansatz. """
        self.set_var_params(var_params)

        self.circuit = Circuit(n_qubits=self.n_spinorbitals)
        self.circuit += self.prepare_reference_state()
        adapt_circuit = []
        for op in self.operators:
            adapt_circuit += pauliword_to_circuit(op, 0.1)

        adapt_circuit = Circuit(adapt_circuit)
        if adapt_circuit.size != 0:
            self.circuit += adapt_circuit
        return self.circuit

    def add_operator(self, pauli_tuple):
        """Add a new operator to our circuit"""
        new_operator = Circuit(pauliword_to_circuit(pauli_tuple, 0.1))
        self.circuit += new_operator
        self.operators.append(pauli_tuple)


class ADAPTSolver:
    """ Add string """

    def __init__(self, opt_dict):

        default_backend_options = {"target": "qulacs", "n_shots": None, "noise_model": None}
        default_options = {"molecule": None, "verbose": False,
                           "tol": 1e-3, "max_cycles": 30,
                           "vqe_options": dict(),
                           "backend": default_backend_options}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                # TODO Raise a warning instead, that variable will not be used unless user made mods to code
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None

    @property
    def operators(self):
        return self.ansatz.operators

    def build(self):

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles
        self.vqe_options['molecule'] = self.molecule
        self.vqe_solver = VQESolver(self.vqe_options)
        self.vqe_solver.build()

        # Initialize ansatz with molecular information
        self.n_spinorbitals = 2*len(self.vqe_solver.mean_field.mo_occ)
        self.ansatz = AdaptAnsatz(n_spinorbitals=self.n_spinorbitals, n_electrons=self.molecule.nelectron)
        self.vqe_solver.ansatz = self.ansatz
        self.ansatz.build_circuit()

        # Initialize pool of operators to draw from during the ADAPT procedure
        # self.pool_operators, self.pool_tuples = get_pool(self.vqe_solver.qubit_hamiltonian, self.n_spinorbitals)
        # self.pool_operators = [commutator(self.vqe_solver.qubit_hamiltonian, element) for element in self.pool_generators]
        # Initialize pool of operators like in ADAPT-VQE paper
        from qsdk.toolboxes.ansatz_generator._unitary_cc import uccsd_singlet_generator, uccsd_singlet_paramsize
        from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
        import math
        n_amplitudes = uccsd_singlet_paramsize(self.n_spinorbitals, self.molecule.nelectron)
        n_singles = int(math.sqrt(2*n_amplitudes + 2.25) - 1.5)
        n_doubles = n_amplitudes - n_singles
        f_op = uccsd_singlet_generator([1.]*n_amplitudes, self.n_spinorbitals, self.molecule.nelectron)
        from copy import deepcopy
        f_op_tmp = deepcopy(f_op)
        self.pool_operators = list()
        for k, v in f_op.terms.items():
            f_op_tmp.terms = {k: v}
            self.pool_operators += [fermion_to_qubit_mapping(f_op_tmp, self.vqe_solver.qubit_mapping,
                                                             self.n_spinorbitals, self.molecule.nelectron)]



        pass

    def simulate(self):
        """ Fill in    """

        operators, energies = list(), list()
        params = np.array([0.0])
        converged = False
        n_cycles = 0

        while n_cycles < self.max_cycles:

            n_cycles += 1
            pool_select = rank_pool(self.pool_operators, self.vqe_solver.ansatz.circuit,
                                    backend=self.vqe_solver.backend, tolerance=self.tol)

            if pool_select > -1:
                operators.append(self.pool_tuples[pool_select])
                self.vqe_solver.ansatz.add_operator(operators[-1])
                self.vqe_solver.initial_var_params = params
                opt_energy, opt_params = self.vqe_solver.simulate()
                params = np.concatenate([opt_params, np.array([0.])])
                energies.append(opt_energy)
            else:
                break

        return energies, operators, self.vqe_solver

    def get_resources(self):
        """ Return resources currently used in underlying VQE """
        return self.vqe_solver.get_resources()


def LBFGSB_optimizer(func, var_params):
    result = minimize(func, var_params, method="L-BFGS-B",
                      options={"disp": False, "maxiter": 100, 'gtol': 1e-10, 'iprint': -1})

    # self.optimal_var_params = result.x
    # self.optimal_energy = result.fun
    # # self.ansatz.build_circuit(self.optimal_var_params)
    # # self.optimal_circuit = self.ansatz.circuit
    #
    # if self.verbose:
    #     print(f"\t\tOptimal VQE energy: {self.optimal_energy}")
    #     print(f"\t\tOptimal VQE variational parameters: {self.optimal_var_params}")
    #     print(f"\t\tNumber of Function Evaluations : {result.nfev}")

    print(result.fun)
    return result.fun, result.x
