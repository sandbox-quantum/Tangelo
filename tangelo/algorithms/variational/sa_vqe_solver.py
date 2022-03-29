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

"""Implements the state-averaged variational quantum eigensolver. Also known as the subspace-search variational quantum
eigensolver.

Ref:
[1] Saad Yalouz, Bruno Senjean, Jakob Gunther, Francesco Buda, Thomas E. O'Brien, Lucas Visscher, "A state-averaged
orbital-optimized hybrid quantum-classical algorithm for a democratic description of ground and excited states",
2021, Quantum Sci. Technol. 6 024004
[2] Ken M Nakanishi, Kosuke Mitarai, Keisuke Fujii, "Subspace-search variational quantum eigensolver for excited states",
Phys. Rev. Research 1, 033062 (2019)
"""

import warnings
import itertools
from copy import deepcopy

import numpy as np

from tangelo.linq import Simulator, Circuit
from tangelo.toolboxes.operators import qubitop_to_qubitham
from tangelo.toolboxes.qubit_mappings import statevector_mapping
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.ansatz_generator import UCCSD, HEA, UpCCGSD, VariationalCircuitAnsatz, UCCGD
from tangelo.toolboxes.ansatz_generator.penalty_terms import combined_penalty
from tangelo.algorithms.variational import BuiltInAnsatze, VQESolver


class SA_VQESolver(VQESolver):
    r"""Solve the electronic structure problem for a molecular system by using
    the state-averaged variational quantum eigensolver (SA-VQE) algorithm.

    This algorithm evaluates the energy of a molecular system by performing
    classical optimization over a parametrized ansatz circuit for multiple reference states.

    Users must first set the desired options of the SA_VQESolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the energy_estimation, simulate, or get_rdm methods. In particular, simulate
    runs the VQE algorithm, returning the optimal energy found by the classical
    optimizer.

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        ansatz (Ansatze) : one of the supported ansatze.
        optimizer (function handle): a function defining the classical optimizer and its behavior.
        initial_var_params (str or array-like) : initial value for the classical optimizer.
        backend_options (dict) : parameters to build the tangelo.linq Simulator class.
        penalty_terms (dict): parameters for penalty terms to append to target qubit Hamiltonian (see penalty_terms
            for more details).
        ansatz_options (dict): parameters for the given ansatz (see given ansatz file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down.
            Default, False has alternating spin up/down ordering.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        verbose (bool): Flag for VQE verbosity.
        ref_states (list): The vector occupations of the reference configurations
        weights (array): The weights of the occupations
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "qubit_mapping": "jw", "ansatz": BuiltInAnsatze.UCCGD,
                           "optimizer": self._default_optimizer,
                           "initial_var_params": None,
                           "backend_options": default_backend_options,
                           "penalty_terms": None,
                           "ansatz_options": dict(),
                           "up_then_down": False,
                           "qubit_hamiltonian": None,
                           "verbose": False,
                           "ref_states": None,
                           "weights": None}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                raise KeyError(f"Keyword :: {k}, not available in VQESolver")

        if self.ref_states is None:
            raise ValueError(f"ref_states must be provided when instantiating {self.__class__.__name__}")
        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

        self.initial_qubit_hamiltonian = deepcopy(self.qubit_hamiltonian) if self.qubit_hamiltonian is not None else None
        self.optimal_energy = None
        self.optimal_var_params = None
        self.builtin_ansatze = set([BuiltInAnsatze.UpCCGSD, BuiltInAnsatze.UCCGD, BuiltInAnsatze.HEA, BuiltInAnsatze.UCCSD])

    def build(self):
        """Build the underlying objects required to run the SA-VQE algorithm
        afterwards.
        """

        if isinstance(self.ansatz, Circuit):
            self.ansatz = VariationalCircuitAnsatz(self.ansatz)

        # Building VQE with a molecule as input.
        if self.molecule:

            # Compute qubit hamiltonian for the input molecular system
            qubit_op = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                mapping=self.qubit_mapping,
                                                n_spinorbitals=self.molecule.n_active_sos,
                                                n_electrons=self.molecule.n_active_electrons,
                                                up_then_down=self.up_then_down,
                                                spin=self.molecule.spin)

            self.core_constant, self.oneint, self.twoint = self.molecule.get_active_space_integrals()

            self.qubit_hamiltonian = qubitop_to_qubitham(qubit_op, self.qubit_mapping, self.up_then_down)

            if self.penalty_terms:
                pen_ferm = combined_penalty(self.molecule.n_active_mos, self.penalty_terms)
                pen_qubit = fermion_to_qubit_mapping(fermion_operator=pen_ferm,
                                                     mapping=self.qubit_mapping,
                                                     n_spinorbitals=self.molecule.n_active_sos,
                                                     n_electrons=self.molecule.n_active_electrons,
                                                     up_then_down=self.up_then_down,
                                                     spin=self.molecule.spin)
                pen_qubit = qubitop_to_qubitham(pen_qubit, self.qubit_hamiltonian.mapping, self.qubit_hamiltonian.up_then_down)
                self.qubit_hamiltonian += pen_qubit

            # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
            if isinstance(self.ansatz, BuiltInAnsatze):
                if self.ansatz == BuiltInAnsatze.UCCSD:
                    self.ansatz = UCCSD(self.molecule, self.qubit_mapping, self.up_then_down, self.molecule.spin)
                elif self.ansatz == BuiltInAnsatze.HEA:
                    self.ansatz_options["reference_state"] = "zero"
                    self.ansatz = HEA(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.UpCCGSD:
                    self.ansatz = UpCCGSD(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.UCCGD:
                    self.ansatz = UCCGD(self.molecule, self.qubit_mapping, up_then_down=self.up_then_down)
                else:
                    raise ValueError(f"Unsupported ansatz for SA_VQESolver. Built-in ansatze:\n\t{self.builtin_ansatze}")
            elif not isinstance(self.ansatz, Ansatz):
                print(type(self.ansatz))
                raise TypeError(f"Invalid ansatz dataype. Expecting instance of Ansatz class, or one of built-in options:\n\t{self.builtin_ansatze}")

        # Building with a qubit Hamiltonian.
        elif self.ansatz in [BuiltInAnsatze.HEA]:
            if self.ansatz == BuiltInAnsatze.HEA:
                self.ansatz = HEA(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
        elif not isinstance(self.ansatz, Ansatz):
            raise TypeError(f"Invalid ansatz dataype. Expecting a custom Ansatz (Ansatz class).")

        self.ansatz.default_reference_state = "zero"
        self.reference_circuits = list()
        for ref_state in self.ref_states:
            vec_to_map = np.concatenate((ref_state[::2], ref_state[1::2])) if self.up_then_down else ref_state
            if self.qubit_mapping.lower() == "scbk":
                mapped_state = statevector_mapping.do_scbk_transform(vec_to_map, len(ref_state))
            elif self.qubit_mapping.lower() == "bk":
                mapped_state = statevector_mapping.do_bk_transform(vec_to_map)
            else:
                mapped_state = vec_to_map
            self.reference_circuits.append(statevector_mapping.vector_to_circuit(mapped_state, self.qubit_mapping))
            self.reference_circuits[-1].name = str(ref_state)

        self.n_states = len(self.ref_states)
        if self.weights is None:
            self.weights = np.ones(self.n_states)/self.n_states
        else:
            if len(self.weights) != self.n_states:
                raise ValueError("Number of elements in weights must equal the number of ref_configs")
            self.weights = np.array(self.weights)
            self.weights = self.weights/sum(self.weights)

        # Set ansatz initial parameters (default or use input), build corresponding ansatz circuit
        self.initial_var_params = self.ansatz.set_var_params(self.initial_var_params)
        self.ansatz.build_circuit()

        # Quantum circuit simulation backend options
        self.backend = Simulator(target=self.backend_options["target"], n_shots=self.backend_options["n_shots"],
                                 noise_model=self.backend_options["noise_model"])

    def simulate(self):
        """Run the VQE algorithm, using the ansatz, classical optimizer, initial
        parameters and hardware backend built in the build method for each reference state.
        """
        if not (self.ansatz and self.backend):
            raise RuntimeError("No ansatz circuit or hardware backend built. Have you called VQESolver.build ?")
        optimal_energy, optimal_var_params = self.optimizer(self.energy_estimation, self.initial_var_params)

        self.optimal_var_params = optimal_var_params
        self.optimal_energy = optimal_energy
        self.ansatz.build_circuit(self.optimal_var_params)
        self.optimal_circuit = self.ansatz.circuit
        return self.optimal_energy

    def energy_estimation(self, var_params):
        """Estimate state-averaged energy using the given ansatz, qubit hamiltonian and compute
        backend. Keeps track of optimal energy and variational parameters along
        the way.

        Args:
             var_params (numpy.array or str): variational parameters to use for
                VQE energy evaluation.

        Returns:
             float: energy computed by VQE using the ansatz and input
                variational parameters.
        """

        # Update variational parameters, compute energy using the hardware backend
        self.ansatz.update_var_params(var_params)
        energy = 0
        self.state_energies = list()
        for i, reference_circuit in enumerate(self.reference_circuits):
            state_energy = self.backend.get_expectation_value(self.qubit_hamiltonian, reference_circuit + self.ansatz.circuit)
            energy += state_energy*self.weights[i]
            self.state_energies.append(state_energy)

        if self.verbose:
            print(f"\tEnergy = {energy:.7f} ")

        return energy
