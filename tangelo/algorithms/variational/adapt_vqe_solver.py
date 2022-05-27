# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module that defines the ADAPT-VQE algorithm framework. ADAPT-VQE is a
variational approach that builds an ansatz iteratively, until a convergence
criteria or a maximum number of cycles is reached. Each iteration ("cycle")
of ADAPT consists in drawing an operator from a pre-defined operator pool,
selecting the one that impacts the energy the most, growing the ansatz circuit
accordingly, and optimizing the variational parameters using VQE.

Ref:
    Grimsley, H.R., Economou, S.E., Barnes, E. et al.
    An adaptive variational algorithm for exact molecular simulations on a
    quantum computer.
    Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2.
"""

import math
import warnings

from scipy.optimize import minimize
from openfermion import commutator
from openfermion import FermionOperator as ofFermionOperator

from tangelo.toolboxes.operators.operators import FermionOperator, QubitOperator
from tangelo.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator as uccgsd_pool
from tangelo.toolboxes.operators import qubitop_to_qubitham
from tangelo.linq import Circuit, Gate
from tangelo.algorithms.variational.vqe_solver import VQESolver


class ADAPTSolver:
    """ADAPT VQE class. This is an iterative algorithm that uses VQE. Methods
    are defined to rank operators with respect to their influence on the total
    energy.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        tol (float): Maximum gradient allowed for a particular operator  before
            convergence.
        max_cycles (int): Maximum number of iterations for ADAPT.
        pool (func): Function that returns a list of FermionOperator. Each
            element represents excitation/operator that has an effect of the
            total energy.
        pool_args (dict) : The arguments for the pool function. Will be unpacked in
            function call as pool(**pool_args)
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        up_then_down (bool): Spin orbitals ordering.
        n_spinorbitals (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        optimizer (func): Optimization function for VQE minimization.
        backend_options (dict): Backend options for the underlying VQE object.
        verbose (bool): Flag for verbosity of VQE.
        deflation_circuits (list[Circuit]): Deflation circuits to add an
            orthogonalization penalty with.
        deflation_coeff (float): The coefficient of the deflation.
        ref_state (array or Circuit): The reference configuration to use. Replaces HF state
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
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
                           "verbose": False,
                           "ref_state": None,
                           "deflation_circuits": list(),
                           "deflation_coeff": 1}

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
        """Builds the underlying objects required to run the ADAPT-VQE
        algorithm.
        """

        # Building molecule data with a pyscf molecule.
        if self.molecule:

            self.n_spinorbitals = self.molecule.n_active_sos
            self.n_electrons = self.molecule.n_active_electrons
            self.spin = self.molecule.spin

            # Compute qubit hamiltonian for the input molecular system
            qubit_op = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                mapping=self.qubit_mapping,
                                                n_spinorbitals=self.n_spinorbitals,
                                                n_electrons=self.n_electrons,
                                                up_then_down=self.up_then_down,
                                                spin=self.spin)

            self.qubit_hamiltonian = qubitop_to_qubitham(qubit_op, self.qubit_mapping, self.up_then_down)

        # Build / set ansatz circuit.
        ansatz_options = {"mapping": self.qubit_mapping, "up_then_down": self.up_then_down,
                          "reference_state": "HF" if self.ref_state is None else "zero"}
        self.ansatz = ADAPTAnsatz(self.n_spinorbitals, self.n_electrons, ansatz_options)

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles.
        self.vqe_options = {"qubit_hamiltonian": self.qubit_hamiltonian,
                            "ansatz": self.ansatz,
                            "optimizer": self.optimizer,
                            "backend_options": self.backend_options,
                            "deflation_circuits": self.deflation_circuits,
                            "deflation_coeff": self.deflation_coeff,
                            "ref_state": self.ref_state
                            }

        self.vqe_solver = VQESolver(self.vqe_options)
        self.vqe_solver.build()

        # If applicable, give vqe_solver access to molecule object
        if self.molecule:
            self.vqe_solver.molecule = self.molecule

        # Getting the pool of operators for the ansatz. If more functionalities
        # are added, this part must be modified and generalized.
        if self.pool_args is None:
            if self.pool == uccgsd_pool:
                self.pool_args = {"n_qubits": self.n_spinorbitals}
            else:
                raise KeyError('pool_args must be defined if using own pool function')

        # Check if pool function returns a QubitOperator or FermionOperator and populate variables
        pool_list = self.pool(**self.pool_args)
        if isinstance(pool_list[0], QubitOperator):
            self.pool_type = 'qubit'
            self.pool_operators = pool_list
        elif isinstance(pool_list[0], (FermionOperator, ofFermionOperator)):
            self.pool_type = 'fermion'
            self.fermionic_operators = pool_list
            self.pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                            mapping=self.qubit_mapping,
                                                            n_spinorbitals=self.n_spinorbitals,
                                                            n_electrons=self.n_electrons,
                                                            up_then_down=self.up_then_down,
                                                            spin=self.spin) for fi in self.fermionic_operators]
        else:
            raise ValueError('pool function must return either QubitOperator or FermionOperator')

        # Cast all coefs to floats (rotations angles are real).
        for qubit_op in self.pool_operators:
            for term, coeff in qubit_op.terms.items():
                qubit_op.terms[term] = math.copysign(1., coeff.imag)

        # Getting commutators to compute gradients:
        # \frac{\partial E}{\partial \theta_n} = \langle \psi | [\hat{H}, A_n] | \psi \rangle
        self.pool_commutators = [commutator(self.qubit_hamiltonian.to_qubitoperator(), element) for element in self.pool_operators]

    def simulate(self):
        """Performs the ADAPT cycles. Each iteration, a VQE minimization is
        done.
        """

        params = self.vqe_solver.ansatz.var_params

        # Construction of the ansatz. self.max_cycles terms are added, unless
        # all operator gradients are less than self.tol.
        while self.iteration < self.max_cycles:
            self.iteration += 1
            if self.verbose:
                print(f"Iteration {self.iteration} of ADAPT-VQE.")

            full_circuit = (self.vqe_solver.ansatz.circuit if self.ref_state is None else
                            self.vqe_solver.reference_circuit + self.vqe_solver.ansatz.circuit)
            pool_select = self.rank_pool(self.pool_commutators, full_circuit,
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
            pool_commutators (QubitOperator): Commutator [H, operator] for each
                generator.
            circuit (tangelo.linq.Circuit): Circuit for measuring each commutator.
            backend (tangelo.linq.Simulator): Backend to compute expectation values.
            tolerance (float): Minimum value for gradient to be considered.

        Returns:
            int: Index of the operators with the highest gradient. If it is not
                bigger than tolerance, returns -1.
        """

        gradient = [abs(backend.get_expectation_value(element, circuit)) for element in pool_commutators]
        for deflate_circuit in self.deflation_circuits:
            for i, pool_op in enumerate(self.pool_operators):
                op_circuit = Circuit([Gate(op[1], op[0]) for tuple in pool_op.terms for op in tuple])
                pool_over = deflate_circuit.inverse() + op_circuit + circuit
                f_dict, _ = backend.simulate(pool_over)
                grad = f_dict.get("0"*self.vqe_solver.ansatz.circuit.width, 0)
                pool_over = deflate_circuit.inverse() + circuit
                f_dict, _ = backend.simulate(pool_over)
                gradient[i] += self.deflation_coeff * grad * f_dict.get("0"*self.vqe_solver.ansatz.circuit.width, 0)
        max_partial = max(gradient)

        if self.verbose:
            print(f"LARGEST PARTIAL DERIVATIVE: {max_partial :4E} \t[{gradient.index(max_partial)}]")

        return gradient.index(max_partial) if max_partial >= tolerance else -1

    def get_resources(self):
        """Returns resources currently used in underlying VQE."""

        return self.vqe_solver.get_resources()

    def LBFGSB_optimizer(self, func, var_params):
        """Default optimizer for ADAPT-VQE."""

        result = minimize(func, var_params, method="L-BFGS-B",
                          options={"disp": False, "maxiter": 100, "gtol": 1e-10, "iprint": -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun

        # Reconstructing the optimal circuit at the end of the ADAPT iterations
        # or when the algorithm has converged.
        if self.converged or self.iteration == self.max_cycles:
            self.ansatz.build_circuit(self.optimal_var_params)
            self.optimal_circuit = (self.vqe_solver.ansatz.circuit if self.ref_state is None else
                                    self.vqe_solver.reference_circuit + self.vqe_solver.ansatz.circuit)

        if self.verbose:
            print(f"VQESolver optimization results:")
            print(f"\tOptimal VQE energy: {result.fun}")
            print(f"\tOptimal VQE variational parameters: {result.x}")
            print(f"\tNumber of Iterations : {result.nit}")
            print(f"\tNumber of Function Evaluations : {result.nfev}")
            print(f"\tNumber of Gradient Evaluations : {result.njev}")

        return result.fun, result.x
