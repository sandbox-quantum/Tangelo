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

"""This module defines the hardware efficient ansatz class, for use in applying
VQE as first defined in "Hardware-efficient Variational Quantum Eigensolver for
Small Molecules and Quantum Magnets" https://arxiv.org/abs/1704.05018.
"""

import numpy as np

from .ansatz import Ansatz
from ._hea_circuit import construct_hea_circuit
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.linq import Circuit


class HEA(Ansatz):
    """This class implements the HEA ansatz. A molecule or a number of qubits +
    a number of electrons can be passed to this class.

    Args:
        molecule (SecondQuantizedMolecule) : The molecular system.
        mapping (str) : one of the supported qubit mapping identifiers. Default:
            "JW"
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False has alternating
            spin up/down ordering.
        n_layers (int): The number of HEA ansatz layers to use. One layer is
            hea_rot_type + grid of CNots. Default: 2.
        rot_type (str): "euler": RzRxRz on each qubit for each rotation layer.
            "real": Ry on each qubit for each rotation layer. Default: "euler".
        n_qubits (int) : The number of qubits in the ansatz.
            Default, None.
        n_electrons (int) : Self-explanatory.
        reference_state (str): "HF": Hartree-Fock reference state. "zero": for
            no reference state. Default: "HF".
        """

    def __init__(self, molecule=None, mapping="jw", up_then_down=False,
                n_layers=2, rot_type="euler", n_qubits=None, n_electrons=None,
                reference_state="HF"):

        if not (bool(molecule) ^ (bool(n_qubits) and (bool(n_electrons) | reference_state == "zero"))):
            raise ValueError(f"A molecule OR qubit + electrons number must be provided when instantiating the HEA.")

        if n_qubits:
            self.n_qubits = n_qubits
            self.n_electrons = n_electrons
        else:
            self.n_qubits = get_qubit_number(mapping, molecule.n_active_sos)
            self.n_electrons = molecule.n_active_electrons

        self.qubit_mapping = mapping
        self.up_then_down = up_then_down
        self.n_layers = n_layers
        self.rot_type = rot_type
        self.reference_state = reference_state

        # Each euler rotation layer has 3 variational parameters per qubit, and one non-variational entangler
        # Each real rotation layer has 1 variational parameter per qubit, and one non-variational entangler
        # There is an additional rotation layer with no entangler.
        if self.rot_type == "euler":
            self.n_var_params = self.n_qubits * 3 * (self.n_layers + 1)
        elif self.rot_type == "real":
            self.n_var_params = self.n_qubits * 1 * (self.n_layers + 1)
        else:
            raise ValueError("Supported rot_type is 'euler'' and 'real'")

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF", "zero"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "zeros"}

        # Default initial parameters for initialization
        self.var_params_default = "random"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros, random numbers,
        MP2 (...), providing some keywords for users, and also supporting direct
        user input (list or numpy array). Return the parameters so that
        workflows such as VQE can retrieve these values.
        """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            else:
                if var_params == "ones":
                    initial_var_params = np.ones((self.n_var_params,), dtype=float)
                elif var_params == "random":
                    initial_var_params = 4 * np.pi * (np.random.random((self.n_var_params,)) - 0.5)
                elif var_params == "zeros":
                    initial_var_params = np.zeros((self.n_var_params,), dtype=float)
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Prepare a circuit generating the HF reference state."""
        if self.reference_state not in self.supported_reference_state:
            raise ValueError(f"{self.reference_state} not in supported reference state methods of:{self.supported_reference_state}")

        if self.reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_qubits,
                                         n_electrons=self.n_electrons,
                                         mapping=self.qubit_mapping,
                                         up_then_down=self.up_then_down)
        elif self.reference_state == "zero":
            return get_reference_circuit(n_spinorbitals=self.n_qubits,
                                         n_electrons=0,
                                         mapping=self.qubit_mapping,
                                         up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""
        self.var_params = self.set_var_params(var_params)

        reference_state_circuit = self.prepare_reference_state()

        hea_circuit = construct_hea_circuit(self.n_qubits, self.n_layers, self.rot_type)

        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + hea_circuit
        else:
            self.circuit = hea_circuit

        self.update_var_params(self.var_params)
        return self.circuit

    def update_var_params(self, var_params):
        """Update variational parameters (done repeatedly during VQE)."""
        self.set_var_params(var_params)
        var_params = self.var_params

        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]
