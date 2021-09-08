"""Module that defines the ADAPT-VQE algorithm framework. ADAPT-VQE is a
variational  approach that builds an ansatz iteratively, until a convergence
criteria or a maximum number of cycles is reached. Each iteration ("cycle")
of ADAPT consists in drawing an operator from a pre-defined operator pool,
selecting the one that impacts the energy the most, growing the ansatz circuit
accordingly, and optimizing the variational parameters using VQE.

Ref:
    Grimsley, H.R., Economou, S.E., Barnes, E. et al.
    An adaptive variational algorithm for exact molecular simulations on a quantum computer.
    Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2
"""

import math
from openfermion import commutator
from openfermion import QubitOperator as ofQubitOperator, FermionOperator as ofFermionOperator
from qsdk.toolboxes.operators.operators import FermionOperator, QubitOperator
from scipy.optimize import minimize
import warnings

from qsdk.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from qsdk.electronic_structure_solvers.vqe_solver import VQESolver
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator as uccgsd_pool
from qsdk.toolboxes.operators import qubitop_to_qubitham


class ADAPTSolver:
    """ADAPT VQE class. This is an iterative algorithm that uses VQE. Methods are
    defined to rank operators with respect to their influence on the total energy.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        mean-field (optional): Mean-field of the molecular system.
        tol (float): Maximum gradient allowed for a particular operator  before
            convergence.
        max_cycles (int): Maximum number of iterations for ADAPT.
        pool (func): Function that returns a list of FermionOperator. Each element
            represents excitation/operator that has an effect of the total
            energy.
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        up_then_down (bool): Spin orbitals ordering.
        n_spinorbitals (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        optimizer (func): Optimization function for VQE minimization.
        backend_options (dict): Backend options for the underlying VQE object.
        verbose (bool): Flag for verbosity of VQE.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": "qulacs", "n_shots": None, "noise_model": None}
        default_options = {"molecule": None, "verbose": False,
                           "tol": 1e-3, "max_cycles": 15,
                           "pool": uccgsd_pool,
                           "pool_args": None,
                           "frozen_orbitals": "frozen_core",
                           "qubit_mapping": "jw",
                           "qubit_hamiltonian": None,
                           "up_then_down": False,
                           "n_spinorbitals": None,
                           "n_electrons": None,
                           "optimizer": self.LBFGSB_optimizer,
                           "backend_options": default_backend_options,
                           "verbose": True}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                # TODO Raise a warning instead, that variable will not be used unless user made mods to code
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

        if self.qubit_hamiltonian:
            if not (self.n_spinorbitals and self.n_electrons):
                raise ValueError("Expecting the number of spin-orbitals (n_spinorbitals) and the number of electrons (n_electrons) with a qubit_hamiltonian.")

        self.ansatz = None
        self.converged = False
        self.iteration = 0
        self.energies = list()

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None

    @property
    def operators(self):
        if self.ansatz is None:
            warnings.warn("operators: this attribute requires the 'build' method to be called first.")
            return None

        return self.ansatz.operators

    @property
    def ferm_operators(self):
        if self.ansatz is None:
            warnings.warn("ferm_operators: this attribute requires the 'build' method to be called first.")
            return None

        return self.ansatz.ferm_operators

    def build(self):
        """Builds the underlying objects required to run the ADAPT-VQE algorithm. """

        # Building molecule data with a pyscf molecule.
        if self.molecule:

            self.n_spinorbitals = self.molecule.n_active_sos
            self.n_electrons = self.molecule.n_active_electrons

            # Compute qubit hamiltonian for the input molecular system
            qubit_op = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                mapping=self.qubit_mapping,
                                                n_spinorbitals=self.n_spinorbitals,
                                                n_electrons=self.n_electrons,
                                                up_then_down=self.up_then_down)

            self.qubit_hamiltonian = qubitop_to_qubitham(qubit_op, self.qubit_mapping, self.up_then_down)

        # Build / set ansatz circuit.
        ansatz_options = {"mapping": self.qubit_mapping, "up_then_down": self.up_then_down}
        self.ansatz = ADAPTAnsatz(self.n_spinorbitals, self.n_electrons, ansatz_options)

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles.
        self.vqe_options = {"qubit_hamiltonian": self.qubit_hamiltonian,
                            "ansatz": self.ansatz,
                            "optimizer": self.optimizer,
                            "backend_options": self.backend_options
                            }

        self.vqe_solver = VQESolver(self.vqe_options)
        self.vqe_solver.build()

        # Getting the pool of operators for the ansatz. If more functionalities
        # are added, this part must be modified and generalized.
        if self.pool_args is None:
            if self.pool == uccgsd_pool:
                self.pool_args = (self.n_spinorbitals,)
            else:
                raise KeyError('pool_args must be defined if using own pool function')
                # Check if pool function returns a QubitOperator or FermionOperator
        if self.pool != uccgsd_pool:
            pool_item = self.pool(*self.pool_args)[0]
            if isinstance(pool_item, (QubitOperator, ofQubitOperator)):
                self.pool_type = 'qubit'
            elif isinstance(pool_item, (FermionOperator, ofFermionOperator)):
                self.pool_type = 'fermion'
            else:
                raise ValueError('pool function must return either QubitOperator or FermionOperator')
        if self.pool_type == 'fermion':
            self.fermionic_operators = self.pool(*self.pool_args)
            self.pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                            mapping=self.qubit_mapping,
                                                            n_spinorbitals=self.n_spinorbitals,
                                                            n_electrons=self.n_electrons,
                                                            up_then_down=self.up_then_down) for fi in self.fermionic_operators]
        else:
            self.pool_operators = self.pool(*self.pool_args)

        # Cast all coefs to floats (rotations angles are real).
        for qubit_op in self.pool_operators:
            for term, coeff in qubit_op.terms.items():
                qubit_op.terms[term] = math.copysign(1., coeff.imag)

        # Getting commutators to compute gradients:
        # \frac{\partial E}{\partial \theta_n} = \langle \psi | [\hat{H}, A_n] | \psi \rangle
        self.pool_commutators = [commutator(self.qubit_hamiltonian.to_qubitoperator(), element) for element in self.pool_operators]

    def simulate(self):
        """Performs the ADAPT cycles. Each iteration, a VQE minimization is done. """

        params = self.vqe_solver.ansatz.var_params

        # Construction of the ansatz. self.max_cycles terms are added, unless
        # all operator gradients are less than self.tol.
        while self.iteration < self.max_cycles:
            self.iteration += 1
            print(f"Iteration {self.iteration} of ADAPT-VQE.")

            pool_select = self.rank_pool(self.pool_commutators, self.vqe_solver.ansatz.circuit,
                                         backend=self.vqe_solver.backend, tolerance=self.tol)

            # If pool selection returns an operator that changes the energy by
            # more than self.tol. Else, the loop is complete and the energy is
            # considered as converged.
            if pool_select > -1:

                # Adding a new operator + initializing its parameters to 0.
                # Previous parameters are kept as they were.
                params += [0.]
                if self.pool_type == 'fermion':
                    self.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select], self.fermionic_operators[pool_select])
                else:
                    self.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select])
                self.vqe_solver.initial_var_params = params

                # Performs a VQE simulation and append the energy to a list.
                # Also, forcing params to be a list to make it easier to append
                # new parameters. The behavior with a np.array is multiplication
                # with broadcasting (not wanted).
                self.vqe_solver.simulate()
                opt_energy = self.vqe_solver.optimal_energy
                params = list(self.vqe_solver.optimal_var_params)
                self.energies.append(opt_energy)
            else:
                self.converged = True
                break

        return self.energies[-1]

    def rank_pool(self, pool_commutators, circuit, backend, tolerance=1e-3):
        """Rank pool of operators with a specific circuit.

        Args:
            pool_commutators (QubitOperator): Commutator [H, operator] for each generator.
            circuit (agnostic_simulator.Circuit): Circuit for measuring each commutator.
            backend (angostic_simulator.Simulator): Backend to measure expectation values.
            tolerance (float): Minimum value for gradient to be considered.

        Returns:
            int: Index of the operators with the highest gradient. If it is not
                bigger than tolerance, returns -1.
        """

        gradient = [abs(backend.get_expectation_value(element, circuit)) for element in pool_commutators]
        max_partial = max(gradient)

        if self.verbose:
            print(f'LARGEST PARTIAL DERIVATIVE: {max_partial :4E} \t[{gradient.index(max_partial)}]')

        return gradient.index(max_partial) if max_partial >= tolerance else -1

    def get_resources(self):
        """Returns resources currently used in underlying VQE. """

        return self.vqe_solver.get_resources()

    def LBFGSB_optimizer(self, func, var_params):
        """Default optimizer for ADAPT-VQE. """

        result = minimize(func, var_params, method="L-BFGS-B",
                          options={"disp": False, "maxiter": 100, 'gtol': 1e-10, 'iprint': -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun

        # Reconstructing the optimal circuit at the end of the ADAPT iterations
        # or when the algorithm has converged.
        if self.converged or self.iteration == self.max_cycles:
            self.ansatz.build_circuit(self.optimal_var_params)
            self.optimal_circuit = self.vqe_solver.ansatz.circuit

        if self.verbose:
            print(f"\t\tOptimal VQE energy: {self.optimal_energy}")
            print(f"\t\tOptimal VQE variational parameters: {self.optimal_var_params}")
            print(f"\t\tNumber of Function Evaluations : {result.nfev}")

        return result.fun, result.x
