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

"""Module that defines the Quantum Imaginary Time Algorithm (QITE)
"""
from copy import copy
import math

from openfermion import FermionOperator as ofFermionOperator
import numpy as np

from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize
from tangelo.toolboxes.operators.operators import FermionOperator, QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator as uccgsd_pool
from tangelo.toolboxes.operators import qubitop_to_qubitham
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.linq import Circuit, Simulator


class QITESolver:
    """QITE class. This is an iterative algorithm that obtains a unitary operator
    that approximates the imaginary time evolution of an initial state.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        dt (float): The imaginary time step size
        min_de (float): Maximum energy change allowed before convergence.
        max_cycles (int): Maximum number of iterations of QITE.
        pool (func): Function that returns a list of FermionOperator. Each
            element represents an excitation/operator that has an effect on the
            total energy.
        pool_args (tuple) : The arguments for the pool function given as a
            tuple.
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        up_then_down (bool): Spin orbitals ordering.
        n_spinorbitals (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        backend_options (dict): Backend options for the underlying QITE propagation
        verbose (bool): Flag for verbosity of QITE.
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
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError("A molecule OR qubit Hamiltonian object must be provided when instantiating "
                             f"{self.__class__.__name__}.")

        if self.qubit_hamiltonian:
            if not (self.n_spinorbitals and self.n_electrons):
                raise ValueError("Expecting the number of spin-orbitals (n_spinorbitals) and the number of "
                                 "electrons (n_electrons) with a qubit_hamiltonian.")

        self.iteration = 0
        self.energies = list()

        self.circuit_list = list()
        self.final_energy = None
        self.final_circuit = None
        self.final_statevector = None

        self.backend = None

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g
        prepare reference wavefunction with HF, multi-reference state, etc).
        These preparations must be consistent with the transform used to obtain
        the qubit operator.
        """

        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                     n_electrons=self.n_electrons,
                                     mapping=self.qubit_mapping,
                                     up_then_down=self.up_then_down,
                                     spin=0)

    def build(self):
        """Builds the underlying objects required to run the QITE algorithm."""

        # Building molecule data with a pyscf molecule.
        if self.molecule:

            self.n_spinorbitals = self.molecule.n_active_sos
            self.n_electrons = self.molecule.n_active_electrons

            # Compute qubit hamiltonian for the input molecular system
            qubit_op = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                mapping=self.qubit_mapping,
                                                n_spinorbitals=self.n_spinorbitals,
                                                n_electrons=self.n_electrons,
                                                up_then_down=self.up_then_down,
                                                spin=0)

            self.qubit_hamiltonian = qubitop_to_qubitham(qubit_op, self.qubit_mapping, self.up_then_down)

        # Getting the pool of operators for the ansatz. If more functionalities
        # are added, this part must be modified and generalized.
        if self.pool_args is None:
            if self.pool == uccgsd_pool:
                self.pool_args = (self.n_spinorbitals,)
            else:
                raise KeyError("pool_args must be defined if using own pool function")
        # Check if pool function returns a QubitOperator or FermionOperator and populate variables
        pool_list = self.pool(*self.pool_args)
        if isinstance(pool_list[0], QubitOperator):
            self.pool_type = "qubit"
            self.full_pool_operators = pool_list
        elif isinstance(pool_list[0], (FermionOperator, ofFermionOperator)):
            self.pool_type = "fermion"
            self.fermionic_operators = pool_list
            self.full_pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                                 mapping=self.qubit_mapping,
                                                                 n_spinorbitals=self.n_spinorbitals,
                                                                 n_electrons=self.n_electrons,
                                                                 up_then_down=self.up_then_down) for fi in self.fermionic_operators]
        else:
            raise ValueError("pool function must return either QubitOperator or FermionOperator")

        # Cast all coefs to floats (rotations angles are real).
        for qubit_op in self.full_pool_operators:
            for term, coeff in qubit_op.terms.items():
                qubit_op.terms[term] = math.copysign(1., coeff.imag)

        # Remove duplicates and only select terms with odd number of Y gates for all mappings except JKMN
        if self.qubit_mapping.upper() != "JKMN":
            reduced_pool_terms = set()
            for qubit_op in self.full_pool_operators:
                for term in qubit_op.terms:
                    count_y = str(term).count("Y")
                    if count_y % 2 == 1:
                        reduced_pool_terms.add(term)
        else:
            reduced_pool_terms = set()
            for qubit_op in self.full_pool_operators:
                for term in qubit_op.terms.keys():
                    if term:
                        reduced_pool_terms.add(term)

        # Generated list of pool_operators and full pool operator.
        self.pool_operators = [QubitOperator(term) for term in reduced_pool_terms]
        self.pool_qubit_op = QubitOperator()
        for term in self.pool_operators:
            self.pool_qubit_op += term

        self.qubit_operator = self.qubit_hamiltonian.to_qubitoperator()

        # Obtain all qubit terms that need to be measured
        self.pool_h = [element*self.qubit_operator for element in self.pool_operators]
        self.pool_pool = [[element1*element2 for element2 in self.pool_operators] for element1 in self.pool_operators]

        # Obtain initial state preparation circuit
        self.circuit_list.append(self.prepare_reference_state())
        self.final_circuit = copy(self.circuit_list[0])

        # Quantum circuit simulation backend options
        self.backend = Simulator(target=self.backend_options["target"], n_shots=self.backend_options["n_shots"],
                                 noise_model=self.backend_options["noise_model"])

        self.use_statevector = self.backend.statevector_available and self.backend._noise_model is None

    def simulate(self):
        """Performs the QITE cycles. Each iteration, a linear system is
        solved to obtain the next unitary. The system to be solved can be found in
        section 3.5 of https://arxiv.org/pdf/2108.04413.pdf

        Returns:
            float: final energy after obtaining running QITE
        """

        # Construction of the circuit. self.max_cycles terms are added, unless
        # the energy change is less than self.min_de.
        if self.use_statevector:
            self.update_statevector(self.backend, self.circuit_list[0])
        self.final_energy = self.energy_expectation(self.backend)
        self.energies.append(self.final_energy)
        while self.iteration < self.max_cycles:
            self.iteration += 1
            if self.verbose:
                print(f"Iteration {self.iteration} of QITE with starting energy {self.final_energy}")

            suv, bu = self.calculate_matrices(self.backend, self.final_energy)

            alphas_array = np.linalg.solve(suv.real, bu.real)
            # convert to dictionary with key as first (only) term of each pool_operator and value self.dt * alphas_array[i]
            alphas_dict = {next(iter(qu_op.terms)): self.dt * alphas_array[i] for i, qu_op in enumerate(self.pool_operators)}
            next_circuit = trotterize(self.pool_qubit_op, alphas_dict, trotter_order=1, n_trotter_steps=1)

            self.circuit_list.append(next_circuit)
            self.final_circuit += next_circuit

            if self.use_statevector:
                self.update_statevector(self.backend, self.circuit_list[self.iteration])

            new_energy = self.energy_expectation(self.backend)
            self.energies.append(new_energy)

            if abs(new_energy - self.final_energy) < self.min_de and self.iteration > 1:
                self.final_energy = new_energy
                break
            self.final_energy = new_energy

        if self.verbose:
            print(f"Final energy of QITE is {self.final_energy}")

        return self.energies[-1]

    def update_statevector(self, backend: Simulator, next_circuit: Circuit):
        r"""Update self.final_statevector by propagating with next_circuit using backend

        Args:
            Simulator: the backend to use for the statevector update
            Circuit: The circuit to apply to the statevector
        """
        _, self.final_statevector = backend.simulate(next_circuit,
                                                     return_statevector=True,
                                                     initial_statevector=self.final_statevector)

    def calculate_matrices(self, backend: Simulator, new_energy: float):
        r"""Calculated matrix elements for imaginary time evolution.
        The matrices are defined in section 3.5 of https://arxiv.org/pdf/2108.04413.pdf

        Args:
            backend (Simulator): the backend from which the matrices are generated
            new_energy (float): the current energy_expectation of the Hamiltonian

        Returns:
            matrix float: The expectation values <\psi| pu^+ pv |\psi>
            array float: The expecation values <\psi| pu^+ H |\psi>
        """

        circuit = Circuit(n_qubits=self.final_circuit.width) if self.use_statevector else self.final_circuit

        ndeltab = np.sqrt(1 - 2 * self.dt * new_energy)
        prefac = -1j/ndeltab
        bu = [prefac*backend.get_expectation_value(element, circuit, initial_statevector=self.final_statevector)
              for element in self.pool_h]
        bu = np.array(bu)
        pool_size = len(self.pool_h)
        suv = np.zeros((pool_size, pool_size), dtype=complex)
        for u in range(pool_size):
            for v in range(u+1, pool_size):
                suv[u, v] = backend.get_expectation_value(self.pool_pool[u][v],
                                                          circuit,
                                                          initial_statevector=self.final_statevector)
                suv[v, u] = suv[u, v]

        return suv, bu

    def energy_expectation(self, backend: Simulator):
        """Estimate energy using the self.final_circuit, qubit hamiltonian and compute
        backend.

        Args:
             backend (Simulator): the backend one computes the energy expectation with

        Returns:
            float: energy computed by the backend
        """
        circuit = Circuit(n_qubits=self.final_circuit.width) if self.use_statevector else self.final_circuit
        energy = backend.get_expectation_value(self.qubit_hamiltonian.to_qubitoperator(),
                                               circuit,
                                               initial_statevector=self.final_statevector)
        return energy

    def get_resources(self):
        """Returns resources currently used in underlying state preparation i.e. self.final_circuit
        the number of pool operators, and the size of qubit_hamiltonian

        Returns:
            dict: Dictionary of various quantum resources required"""
        resources = dict()
        resources["qubit_hamiltonian_terms"] = len(self.qubit_hamiltonian.terms)
        resources["pool_size"] = len(self.pool_operators)
        resources["circuit_width"] = self.final_circuit.width
        resources["circuit_gates"] = self.final_circuit.size
        resources["circuit_2qubit_gates"] = self.final_circuit.counts.get("CNOT", 0)
        return resources
