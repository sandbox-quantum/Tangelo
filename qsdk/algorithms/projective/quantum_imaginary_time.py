# Copyright 2021 1QB Information Technologies Inc.
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

"""Module that defines the quantum imaginary time algorithm
"""

import math
from openfermion import FermionOperator as ofFermionOperator
from qsdk.toolboxes.ansatz_generator.ansatz_utils import trotterize
from qsdk.toolboxes.operators.operators import FermionOperator, QubitOperator
import numpy as np

from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator as uccgsd_pool
from qsdk.toolboxes.operators import qubitop_to_qubitham
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.backendbuddy import Circuit, Simulator


class QITESolver:
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
        pool_args (tuple) : The arguments for the pool function given as a
            tuple.
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        up_then_down (bool): Spin orbitals ordering.
        n_spinorbitals (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        optimizer (func): Optimization function for VQE minimization.
        backend_options (dict): Backend options for the underlying VQE object.
        verbose (bool): Flag for verbosity of VQE.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "dt": 0.5, "max_cycles": 100,
                           "min_de": 1.e-7,
                           "pool": uccgsd_pool,
                           "pool_args": None,
                           "frozen_orbitals": "frozen_core",
                           "qubit_mapping": "jw",
                           "qubit_hamiltonian": None,
                           "up_then_down": False,
                           "n_spinorbitals": None,
                           "n_electrons": None,
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

        self.circuit_list = list()
        self.final_energy = None
        self.final_circuit = None
        self.final_statevector = None

        # Quantum circuit simulation backend options
        self.backend = Simulator(target=self.backend_options["target"], n_shots=self.backend_options["n_shots"],
                                 noise_model=self.backend_options["noise_model"])

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g
        prepare reference wavefunction with HF, multi-reference state, etc).
        These preparations must be consistent with the transform used to obtain
        the qubit operator.
        """

        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                     n_electrons=self.n_electrons,
                                     mapping=self.qubit_mapping,
                                     up_then_down=self.up_then_down)

    def build(self):
        """Builds the underlying objects required to run the ADAPT-VQE
        algorithm.
        """

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

        # Getting the pool of operators for the ansatz. If more functionalities
        # are added, this part must be modified and generalized.
        if self.pool_args is None:
            if self.pool == uccgsd_pool:
                self.pool_args = (self.n_spinorbitals,)
            else:
                raise KeyError('pool_args must be defined if using own pool function')
        # Check if pool function returns a QubitOperator or FermionOperator and populate variables
        pool_list = self.pool(*self.pool_args)
        if isinstance(pool_list[0], QubitOperator):
            self.pool_type = 'qubit'
            self.full_pool_operators = pool_list
        elif isinstance(pool_list[0], (FermionOperator, ofFermionOperator)):
            self.pool_type = 'fermion'
            self.fermionic_operators = pool_list
            self.full_pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                                 mapping=self.qubit_mapping,
                                                                 n_spinorbitals=self.n_spinorbitals,
                                                                 n_electrons=self.n_electrons,
                                                                 up_then_down=self.up_then_down) for fi in self.fermionic_operators]
        else:
            raise ValueError('pool function must return either QubitOperator or FermionOperator')

        # Cast all coefs to floats (rotations angles are real).
        for qubit_op in self.full_pool_operators:
            for term, coeff in qubit_op.terms.items():
                qubit_op.terms[term] = math.copysign(1., coeff.imag)

        reduced_pool_terms = list()
        for qubit_op in self.full_pool_operators:
            for term in qubit_op.terms.keys():
                count_y = 0
                for op in term:
                    if 'Y' in op:
                        count_y += 1
                if count_y % 2 == 1 and term not in reduced_pool_terms:
                    reduced_pool_terms.append(term)

        self.pool_operators = [QubitOperator(term) for term in reduced_pool_terms]
        self.pool_qubit_op = QubitOperator()
        for pool_term in self.pool_operators:
            self.pool_qubit_op += pool_term

        # Getting commutators to compute gradients:
        # \frac{\partial E}{\partial \theta_n} = \langle \psi | [\hat{H}, A_n] | \psi \rangle
        self.pool_h = [element*self.qubit_hamiltonian.to_qubitoperator() for element in self.pool_operators]
        self.pool_pool = [[element1*element2 for element2 in self.pool_operators] for element1 in self.pool_operators]
        self.circuit_list.append(self.prepare_reference_state())
        self.final_circuit = self.circuit_list[0]

    def simulate(self):
        """Performs the QITE cycles. Each iteration, a linear system is
        solved to obtain the next unitary.
        """

        # Construction of the ansatz. self.max_cycles terms are added, unless
        # all operator gradients are less than self.tol.
        self.final_energy = self.energy_expectation(self.backend)
        self.energies.append(self.final_energy)
        while self.iteration < self.max_cycles:
            self.iteration += 1
            print(f"Iteration {self.iteration} of QITE.")

            suv, bu = self.calculate_matrices(self.backend, self.final_energy)

            new_energy = self.energy_expectation(self.backend)
            self.energies.append(new_energy)

            if abs(new_energy - self.final_energy) < self.min_de and self.iteration > 1:
                self.final_energy = new_energy
                break
            else:
                self.final_energy = new_energy

            alphas = self.dt*np.linalg.solve(suv.real, bu.real)
            next_circuit, _ = trotterize(self.pool_qubit_op, alphas, trotter_order=1, num_trotter_steps=1)

            self.circuit_list.append(next_circuit)
            self.final_circuit += next_circuit

        return self.energies[-1]

    def calculate_matrices(self, backend: Simulator, new_energy: float):
        r"""Calculated matrix elements for imaginary time evolution.

        Args:
            backend (Simulator): the backend from which the matrices are generated
            new_energy (float): the current energy_expectation of the Hamiltonian

        Returns:
            suv (matrix float): The expectation values <\psi| pu^+ pv |\psi>
            bu (array float): The expecation values <\psi| pu^+ H |\psi>
        """

        _, self.final_statevector = backend.simulate(self.final_circuit, return_statevector=True)

        ndeltab = np.sqrt(1-2*self.dt*new_energy)
        bu = [backend.get_expectation_value(element,
                                            Circuit(n_qubits=self.final_circuit.width),
                                            initial_statevector=self.final_statevector)/ndeltab for element in self.pool_h]
        bu = -1j*np.array(bu)
        pool_size = len(self.pool_h)
        suv = np.zeros((pool_size, pool_size), dtype=complex)
        for u in range(pool_size):
            suv[u, u] = 0
            for v in range(u+1, pool_size):
                suv[u, v] = backend.get_expectation_value(self.pool_pool[u][v],
                                                          Circuit(n_qubits=self.final_circuit.width),
                                                          initial_statevector=self.final_statevector)
                suv[v, u] = suv[u, v]

        return suv, bu

    def energy_expectation(self, backend: Simulator):
        """Estimate energy using the self.final_circuit, qubit hamiltonian and compute
        backend.

        Args:
             backend (Simulator): the backend one computes the energy expectation with

        Returns:
             energy (float): energy computed by the backend
        """
        energy = backend.get_expectation_value(self.qubit_hamiltonian.to_qubitoperator(),
                                               Circuit(n_qubits=self.final_circuit.width),
                                               initial_statevector=self.final_statevector)
        return energy

    def get_resources(self):
        """Returns resources currently used in underlying VQE."""

        return None
