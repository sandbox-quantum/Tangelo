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

"""This module defines the pUCCD ansatz class. The molecular FermionOperator is
expected to be converted to a BosonOperator (electrons in pairs). Single bosonic
excitations (corresponding to double fermion excitations) form the ansatz. Those
excitations are transformed into a quantum circuit via Givens rotations.

Ref:
    - V.E. Elfving, M. Millaruelo, J.A. Gámez, and C. Gogolin.
        Phys. Rev. A 103, 032605 (2021).
"""

import itertools
import numpy as np

from tangelo.linq import Circuit
from tangelo.toolboxes.ansatz_generator import Ansatz
from tangelo.toolboxes.ansatz_generator.ansatz_utils import givens_gate
from tangelo.toolboxes.qubit_mappings.statevector_mapping import vector_to_circuit


class pUCCD(Ansatz):
    """This class implements the pUCCD ansatz, as described in Phys. Rev. A 103,
    032605 (2021). Electrons are described as hard-core boson and only double
    excitations are considered.

    Args:
        molecule (SecondQuantizedMolecule): Self-explanatory.
        reference_state (string): String refering to an initial state.
            Default: "HF".
    """

    def __init__(self, molecule, reference_state="HF"):

        if molecule.spin != 0:
            raise NotImplementedError("pUCCD is implemented only for closed-shell system.")

        self.molecule = molecule
        self.n_spatial_orbitals = molecule.n_active_mos
        self.n_electrons = molecule.n_active_electrons

        # Set total number of parameters.
        self.n_occupied = int(self.n_electrons / 2)
        self.n_virtual = self.n_spatial_orbitals - self.n_occupied
        self.n_var_params = self.n_occupied * self.n_virtual

        # Supported reference state initialization.
        self.supported_reference_state = {"HF", "zero"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "random"}

        # Default initial parameters for initialization.
        self.var_params_default = "ones"
        self.reference_state = reference_state

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as ones, zeros or random
        numbers providing some keywords for users, and also supporting direct
        user input (list or numpy array). Return the parameters so that
        workflows such as VQE can retrieve these values.
        """
        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            var_params = var_params.lower()
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = 2.e-1 * (np.random.random((self.n_var_params,)) - 0.5)
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g
        prepare reference wavefunction with HF, multi-reference state, etc).
        These preparations must be consistent with the transform used to obtain
        the qubit operator.
        """

        if self.reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.reference_state == "HF":
            vector = [1 if i < self.n_electrons // 2 else 0 for i in range(self.n_spatial_orbitals)]
            return vector_to_circuit(vector)
        elif self.reference_state == "zero":
            return Circuit()

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state
        preparation ansatz (with currently specified initial_state and
        var_params).
        """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        excitations = self._get_double_excitations()

        # Prepend reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Obtain quantum circuit through trivial trotterization of the qubit operator
        # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
        self.exc_to_param_mapping = dict()
        rotation_gates = []

        # Parallel ordering (rotations on different qubits can happen at the
        # same time.
        excitations_per_layer = [[]]
        free_qubits_per_layer = [set(range(self.n_spatial_orbitals))]

        # Classify excitations into circuit layers (single pass on all
        # excitations).
        for p, q in excitations:
            excitations_added = False
            for qubit_indices, gates in zip(free_qubits_per_layer, excitations_per_layer):
                if p in qubit_indices and q in qubit_indices:
                    qubit_indices -= {p, q}
                    gates += [(p, q)]
                    excitations_added = True
                    break

            # If the excitation cannot be added to at least one previous layer,
            # create a new layer.
            if not excitations_added:
                excitations_per_layer.append([(p, q)])
                qubit_indices = set(range(self.n_spatial_orbitals))
                qubit_indices -= {p, q}
                free_qubits_per_layer.append(qubit_indices)

        excitations = list(itertools.chain.from_iterable(excitations_per_layer))
        self.exc_to_param_mapping = {v: k for k, v in enumerate(excitations)}

        rotation_gates = [givens_gate((p, q), 0., is_variational=True) for (p, q) in excitations]
        rotation_gates = list(itertools.chain.from_iterable(rotation_gates))

        puccd_circuit = Circuit(rotation_gates)

        # Skip over the reference state circuit if it is empty.
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + puccd_circuit
        else:
            self.circuit = puccd_circuit

        self.update_var_params(self.var_params)
        return self.circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the already-built
        ansatz circuit member. Preferable to rebuilt your circuit from scratch,
        which can be an involved process.
        """

        self.set_var_params(var_params)
        var_params = self.var_params

        excitations = self._get_double_excitations()
        for i, (p, q) in enumerate(excitations):
            gate_index = self.exc_to_param_mapping[(p, q)]
            self.circuit._variational_gates[gate_index].parameter = var_params[i]

    def _get_double_excitations(self):
        """Construct the UCC double excitations for the given amount of occupied
        and virtual orbitals.

        Returns:
            list of int tuples: List of (p, q) excitations corresponding to the
                occupied orbital p to virtual orbital q.
        """

        # Generate double indices in seniority 0 space.
        excitations = list()
        for p, q in itertools.product(
            range(self.n_occupied),
            range(self.n_occupied, self.n_occupied+self.n_virtual)
        ):
            excitations += [(p, q)]

        return excitations
