# Copyright 2023 Good Chemistry Company.
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
from typing import Optional, Union, List, Callable

from scipy.optimize import minimize
from openfermion import commutator
from openfermion import FermionOperator as ofFermionOperator

from tangelo import SecondQuantizedMolecule
from tangelo.toolboxes.operators.operators import FermionOperator, QubitOperator
from tangelo.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator as uccgsd_pool
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
        pool (func): Function that returns a list of FermionOperator or QubitOperator. Each
            element represents excitation/operator that has an effect of the
            total energy.
        pool_args (dict) : The arguments for the pool function. Will be unpacked in
            function call as pool(**pool_args)
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        up_then_down (bool): Spin orbitals ordering.
        n_spinorbitals (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        spin (int): The spin of the system (# alpha - # beta electrons)
        optimizer (func): Optimization function for VQE minimization.
        backend_options (dict): Backend options for the underlying VQE object.
        simulate_options (dict): Options for fine-control of the simulator backend, including desired measurement results, etc.
        verbose (bool): Flag for verbosity of VQE.
        deflation_circuits (list[Circuit]): Deflation circuits to add an
            orthogonalization penalty with.
        deflation_coeff (float): The coefficient of the deflation.
        projective_circuit (Circuit): A terminal circuit that projects into the correct space, always added to
            the end of the ansatz circuit.
        ref_state (array or Circuit): The reference configuration to use. Replaces HF state
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}

        copt_dict = opt_dict.copy()

        self.molecule: Optional[SecondQuantizedMolecule] = copt_dict.pop("molecule", None)
        self.tol: float = copt_dict.pop("tol", 1e-3)
        self.max_cycles: int = copt_dict.pop("max_cycles", 15)
        self.pool: Callable[..., Union[List[QubitOperator], List[FermionOperator]]] = copt_dict.pop("pool", uccgsd_pool)
        self.pool_args: dict = copt_dict.pop("pool_args", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.optimizer = copt_dict.pop("optimizer", self.LBFGSB_optimizer)
        self.backend_options: dict = copt_dict.pop("backend_options", default_backend_options)
        self.simulate_options: dict = copt_dict.pop("simulate_options", dict())
        self.deflation_circuits: Optional[List[Circuit]] = copt_dict.pop("deflation_circuits", list())
        self.deflation_coeff: float = copt_dict.pop("deflation_coeff", 1.)
        self.up_then_down: bool = copt_dict.pop("up_then_down", False)
        self.spin: int = copt_dict.pop("spin", 0)
        self.qubit_hamiltonian: QubitOperator = copt_dict.pop("qubit_hamiltonian", None)
        self.verbose: bool = copt_dict.pop("verbose", False)
        self.projective_circuit: Circuit = copt_dict.pop("projective_circuit", None)
        self.ref_state: Optional[Union[list, Circuit]] = copt_dict.pop("ref_state", None)
        self.n_spinorbitals: Union[None, int] = copt_dict.pop("n_spinorbitals", None)
        self.n_electrons: Union[None, int] = copt_dict.pop("n_electrons", None)

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

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

        # Building molecule data.
        if self.molecule:

            self.n_spinorbitals = self.molecule.n_active_sos
            self.n_electrons = self.molecule.n_active_electrons
            self.spin = self.molecule.active_spin

            # Compute qubit hamiltonian for the input molecular system
            self.qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                              mapping=self.qubit_mapping,
                                                              n_spinorbitals=self.n_spinorbitals,
                                                              n_electrons=self.n_electrons,
                                                              up_then_down=self.up_then_down,
                                                              spin=self.spin)

        # Build / set ansatz circuit.
        ansatz_options = {"mapping": self.qubit_mapping, "up_then_down": self.up_then_down,
                          "reference_state": "HF" if self.ref_state is None else "zero"}
        self.ansatz = ADAPTAnsatz(self.n_spinorbitals, self.n_electrons, self.spin, ansatz_options)

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles.
        self.vqe_options = {"qubit_hamiltonian": self.qubit_hamiltonian,
                            "ansatz": self.ansatz,
                            "optimizer": self.optimizer,
                            "backend_options": self.backend_options,
                            "simulate_options": self.simulate_options,
                            "deflation_circuits": self.deflation_circuits,
                            "deflation_coeff": self.deflation_coeff,
                            "projective_circuit": self.projective_circuit,
                            "ref_state": self.ref_state
                            }

        self.vqe_solver = VQESolver(self.vqe_options)

        # Circuits without variational parameter raise a warning in VQESolver.
        warnings.filterwarnings("ignore", message="No variational gate found in the circuit.")
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

        # Only a qubit operator is provided with a FermionOperator pool.
        if not (self.n_spinorbitals and self.n_electrons and isinstance(self.spin, int)):
            raise ValueError("Expecting the number of spin-orbitals (n_spinorbitals), "
                             "the number of electrons (n_electrons) and the spin (spin) with "
                             "a qubit_hamiltonian when working with a pool of fermion operators.")

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
        self.pool_commutators = [commutator(self.qubit_hamiltonian, element) for element in self.pool_operators]

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
            if self.projective_circuit:
                full_circuit += self.projective_circuit
            gradients = self.compute_gradients(full_circuit, backend=self.vqe_solver.backend)
            pool_select = self.choose_operator(gradients, tolerance=self.tol)

            # If pool selection returns an operator that changes the energy by
            # more than self.tol. Else, the loop is complete and the energy is
            # considered as converged.
            if pool_select:
                # Adding a new operator + initializing its parameters to 0.
                # Previous parameters are kept as they were.
                params += [0.] * len(pool_select)
                for ps in pool_select:
                    if self.pool_type == 'fermion':
                        self.vqe_solver.ansatz.add_operator(self.pool_operators[ps], self.fermionic_operators[ps])
                    else:
                        self.vqe_solver.ansatz.add_operator(self.pool_operators[ps])
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

        # Reconstructing the optimal circuit at the end of the ADAPT iterations
        # or when the algorithm has converged.
        self.ansatz.build_circuit(self.optimal_var_params)
        self.optimal_circuit = (self.vqe_solver.ansatz.circuit if self.ref_state is None else
                                self.vqe_solver.reference_circuit + self.vqe_solver.ansatz.circuit)

        return self.energies[-1]

    def compute_gradients(self, circuit, backend):
        """Compute gradients for the operator pool with a specific circuit.

        Args:
            circuit (tangelo.linq.Circuit): Circuit for measuring each commutator.
            backend (tangelo.linq.Backend): Backend to compute expectation values.

       Returns:
            list of float: Operator gradients.
        """

        gradient = [abs(backend.get_expectation_value(element, circuit, **self.simulate_options)) for element in self.pool_commutators]
        for deflate_circuit in self.deflation_circuits:
            for i, pool_op in enumerate(self.pool_operators):
                op_circuit = Circuit([Gate(op[1], op[0]) for tuple in pool_op.terms for op in tuple])
                pool_over = deflate_circuit.inverse() + op_circuit + circuit
                f_dict, _ = backend.simulate(pool_over, **self.simulate_options)
                grad = f_dict.get("0"*self.vqe_solver.ansatz.circuit.width, 0)
                pool_over = deflate_circuit.inverse() + circuit
                f_dict, _ = backend.simulate(pool_over, **self.simulate_options)
                gradient[i] += self.deflation_coeff * grad * f_dict.get("0"*self.vqe_solver.ansatz.circuit.width, 0)

        return gradient

    def choose_operator(self, gradients, tolerance=1e-3):
        """Choose next operator to add according to the ADAPT-VQE algorithm.

        Args:
            gradients (list of float): Operator gradients corresponding to
                self.pool_operators.
            tolerance (float): Minimum value for gradient to be considered.

        Returns:
            list of int: Index (list of length=1) of the operator with the
                highest gradient. If it is not bigger than tolerance, returns
                an empty list.
        """
        sorted_op_indices = sorted(range(len(gradients)), key=lambda k: gradients[k])
        max_partial = gradients[sorted_op_indices[-1]]

        if self.verbose:
            print(f"LARGEST PARTIAL DERIVATIVE: {max_partial :4E}")

        return [sorted_op_indices[-1]] if max_partial >= tolerance else []

    def get_resources(self):
        """Returns resources currently used in underlying VQE."""

        return self.vqe_solver.get_resources()

    def LBFGSB_optimizer(self, func, var_params):
        """Default optimizer for ADAPT-VQE."""

        result = minimize(func, var_params, method="L-BFGS-B",
                          options={"disp": False, "maxiter": 100, "gtol": 1e-10, "iprint": -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun

        if self.verbose:
            print(f"VQESolver optimization results:")
            print(f"\tOptimal VQE energy: {result.fun}")
            print(f"\tOptimal VQE variational parameters: {result.x}")
            print(f"\tNumber of Iterations : {result.nit}")
            print(f"\tNumber of Function Evaluations : {result.nfev}")
            print(f"\tNumber of Gradient Evaluations : {result.njev}")

        return result.fun, result.x
