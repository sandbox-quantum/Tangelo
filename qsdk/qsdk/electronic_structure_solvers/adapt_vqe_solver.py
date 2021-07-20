"""ADAPT-VQE algorithm framework, to solve electronic structure calculations.
It consists of constructing the ansatz as VQE iterations are performed.
From a set of operators, the most important one (stepiest energy gradient) versus
the current circuit is chosen. This operator is added to the ansatz, converted into
a set of gates and appended to the circuit. An VQE minimization is performed to
get a set of parameters that minimize the energy. The process is repeated n times
to end hopefully with a good ansatz for the studied molecule (or Hamiltonian).

Ref:
Grimsley, H.R., Economou, S.E., Barnes, E. et al.
An adaptive variational algorithm for exact molecular simulations on a quantum computer.
Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2
"""

from scipy.optimize import minimize
import numpy as np
from openfermion import commutator
import math

from qsdk.electronic_structure_solvers.vqe_solver import Ansatze, VQESolver
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import pauliword_to_circuit
from agnostic_simulator import Circuit
from qsdk.toolboxes.operators import QubitOperator
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz
from qsdk.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core
from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping

from qsdk.toolboxes.ansatz_generator._unitary_cc import uccsd_singlet_generator, uccsd_singlet_paramsize
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from copy import deepcopy


# TODO: put things into the right folders...
# TODO: Generators pool of UCCGSD.

class AdaptUCCGSD(Ansatz):
    def __init__(self, n_spinorbitals, operators=list()):
        pass

class AdaptUCCSD(Ansatz):

    def __init__(self, n_spinorbitals, n_electrons, operators=list()):

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.operators = operators

        self.var_params = None
        self.circuit = None
        self.length_operators = list()
        self.var_params_prefactor = list()

    @property
    def n_var_params(self):
        return len(self.length_operators)

    def set_var_params(self, var_params=None):
        """ Set initial variational parameter values. Defaults to zeros. """
        if var_params is None:
            var_params = np.zeros(self.n_var_params, dtype=float)
        elif len(var_params) != self.n_var_params:
            raise ValueError('Invalid number of parameters.')
        self.var_params = var_params
        return var_params

    def update_var_params(self, var_params):
        """ Update variational parameters (done repeatedly during VQE) """
        for var_index in range(self.n_var_params):
            length_op = self.length_operators[var_index] # 2 or 8

            param_index = sum(self.length_operators[:var_index])
            for param_subindex in range(length_op):
                prefactor = self.var_params_prefactor[param_index+param_subindex]
                self.circuit._variational_gates[param_index+param_subindex].parameter = prefactor*var_params[var_index]

    def prepare_reference_state(self):
        """ Prepare a circuit generating the HF reference state. """
        # TODO non hardcoded mapping
        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping='JW', up_then_down=True)

    def build_circuit(self, var_params=None):
        """ Construct the variational circuit to be used as our ansatz. """
        self.set_var_params(var_params)

        self.circuit = Circuit(n_qubits=self.n_spinorbitals)
        self.circuit += self.prepare_reference_state()
        #adapt_circuit = list()
        #for op in self.operators:
        #    adapt_circuit += pauliword_to_circuit(op, .1)

        #adapt_circuit = Circuit(adapt_circuit)
        #if adapt_circuit.size != 0:
        #    self.circuit += adapt_circuit

        # Must be included because after convergence, the circuit is rebuilt.
        # If this is not present, the gradient on the previous operator chosen
        # is selected again (as the parameters are not minimized in respect to it
        # as with initialized coefficient to 0.1). If the parameters are
        # updated, it stays consistent.
        self.update_var_params(self.var_params)

        return self.circuit

    def add_operator(self, pauli_operator):
        """Add a new operator to our circuit"""

        self.length_operators += [len(pauli_operator.terms)]

        for pauli_term in pauli_operator.get_operators():
            coeff = list(pauli_term.terms.values())[0]
            self.var_params_prefactor += [math.copysign(1., coeff)]

            pauli_tuple = list(pauli_term.terms.keys())[0]
            new_operator = Circuit(pauliword_to_circuit(pauli_tuple, 0.1))

            #self.operators.append(pauli_operator.terms)
            self.circuit += new_operator

    def get_pool(self):
        """TBD
        """
        # Initialize pool of operators like in ADAPT-VQE paper, based on single and double excitations (Work in progress)
        # Use uccsd functions from openfermion to get a hold on the single and second excitations, per Lee's advice.

        n_spatial_orbitals = self.n_spinorbitals // 2
        n_occupied = int(np.ceil(self.n_electrons / 2))
        n_virtual = n_spatial_orbitals - n_occupied
        n_singles = n_occupied * n_virtual
        n_doubles = n_singles * (n_singles + 1) // 2
        n_amplitudes = n_singles + n_doubles

        f_op = uccsd_singlet_generator([1.]*n_amplitudes, 2*n_spatial_orbitals, self.n_electrons)
        lst_fermion_op = list()
        for i in f_op.get_operator_groups(len(f_op.terms) // 2):
            lst_fermion_op.append(i)

        pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                mapping="JW",
                                                n_spinorbitals=self.n_spinorbitals,
                                                n_electrons=self.n_electrons,
                                                up_then_down=True) for fi in lst_fermion_op]

        # Cast all coefs to floats (rotations angles are real)
        for qubit_op in pool_operators:
            for key in qubit_op.terms:
                qubit_op.terms[key] = math.copysign(1., float(qubit_op.terms[key].imag))
            qubit_op.compress()

        return pool_operators, lst_fermion_op


class ADAPTSolver:
    """ ADAPT VQE class.

    Attributes:
        molecule (MolecularData): The molecular system.
        mean-field (optional): mean-field of molecular system.
        frozen_orbitals (list[int]): a list of indices for frozen orbitals.
            Default is the string "frozen_core", corresponding to the output
            of the function molecular_computation.frozen_orbitals.get_frozen_core.
        qubit_mapping (str): one of the supported qubit mapping identifiers
        ansatz (Ansatze): one of the supported ansatze.
        vqe_options:
        verbose (bool): Flag for verbosity of VQE.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": "qulacs", "n_shots": None, "noise_model": None}
        default_options = {"molecule": None, "mean_field": None, "verbose": False,
                           "tol": 1e-3, "max_cycles": 15,
                           "frozen_orbitals": "frozen_core",
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

        # Building molecule data with a pyscf molecule.
        if self.molecule:
            # Build adequate mean-field (RHF for now).
            if not self.mean_field:
                self.mean_field = prepare_mf_RHF(self.molecule)

            # Same default as in vanilla VQE.
            if self.frozen_orbitals == "frozen_core":
                self.frozen_orbitals = get_frozen_core(self.molecule)

            # Compute qubit hamiltonian for the input molecular system.
            self.qemist_molecule = MolecularData(self.molecule, self.mean_field, self.frozen_orbitals)

            self.n_spinorbitals = 2 * self.qemist_molecule.n_orbitals
            self.n_electrons = self.qemist_molecule.n_electrons

            self.fermionic_hamiltonian = self.qemist_molecule.get_molecular_hamiltonian()
            self.qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=self.fermionic_hamiltonian,
                                                              mapping="JW",
                                                              n_spinorbitals=self.n_spinorbitals,
                                                              n_electrons=self.n_electrons,
                                                              up_then_down=True)
        else:
            assert(self.n_spinnorbitals)
            assert(self.n_electrons)

        # Initialize ansatz with molecular information.
        self.ansatz = AdaptUCCSD(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons)

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles
        #self.vqe_options['molecule'] = self.molecule
        self.vqe_options["qubit_hamiltonian"] = self.qubit_hamiltonian
        self.vqe_options["ansatz"] = self.ansatz
        #self.vqe_options["verbose"] =  True
        self.vqe_options["optimizer"] = self.LBFGSB_optimizer
        self.vqe_solver = VQESolver(self.vqe_options)
        self.vqe_solver.build()

        #self.pool_generators, self.pool_tuples = get_pool(self.qubit_hamiltonian, self.n_spinorbitals)
        self.pool_operators, self.fermionic_operators = self.ansatz.get_pool()
        self.pool_commutators = self.get_commutators(self.qubit_hamiltonian, self.pool_operators)

    def simulate(self):
        """ Fill in    """

        operators, energies = list(), list()
        params = list()

        for _ in range(self.max_cycles):
            pool_select = self.rank_pool(self.pool_commutators, self.vqe_solver.ansatz.circuit,
                                    backend=self.vqe_solver.backend, tolerance=self.tol)

            if pool_select > -1:
                # TODO: add print?
                params = list(params) + [0.]

                operators += [self.fermionic_operators[pool_select]]
                self.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select])

                self.vqe_solver.initial_var_params = params
                opt_energy, params = self.vqe_solver.simulate()
                energies.append(opt_energy)
            else:
                break

        return energies, operators

    def get_commutators(self, qubit_hamiltonian, pool_generators):
        gradient_operators = [commutator(qubit_hamiltonian, element) for element in pool_generators]
        return gradient_operators

    def rank_pool(self, pool_commutators, circuit, backend, tolerance=1e-3):
        gradient = [abs(backend.get_expectation_value(element, circuit)) for element in pool_commutators]
        max_partial = max(gradient)
        print(f'LARGEST PARTIAL DERIVATIVE: {max_partial :4E} \t[{gradient.index(max_partial)}]')
        return gradient.index(max_partial) if max_partial >= tolerance else -1

    def get_resources(self):
        """ Return resources currently used in underlying VQE """
        return self.vqe_solver.get_resources()

    def LBFGSB_optimizer(self, func, var_params):
        result = minimize(func, var_params, method="L-BFGS-B",
                        options={"disp": False, "maxiter": 100, 'gtol': 1e-10, 'iprint': -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun
        #self.ansatz.build_circuit(self.optimal_var_params)
        #self.optimal_circuit = self.vqe_solver.ansatz.circuit
        #
        # if self.verbose:
        #     print(f"\t\tOptimal VQE energy: {self.optimal_energy}")
        #     print(f"\t\tOptimal VQE variational parameters: {self.optimal_var_params}")
        #     print(f"\t\tNumber of Function Evaluations : {result.nfev}")

        return result.fun, result.x
