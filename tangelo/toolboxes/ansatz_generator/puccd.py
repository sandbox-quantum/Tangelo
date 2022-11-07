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

"""TBD
V.E. Elfving, M. Millaruelo, J.A. Gámez, and C. Gogolin, Phys. Rev. A 103, 032605 (2021).
"""

import itertools
import numpy as np

from tangelo.linq import Circuit
from tangelo.toolboxes.ansatz_generator import Ansatz
from tangelo.toolboxes.ansatz_generator.ansatz_utils import givens_gate
from tangelo.toolboxes.qubit_mappings.statevector_mapping import vector_to_circuit


class pUCCD(Ansatz):
    """TBD.
    """

    def __init__(self, molecule, reference_state="HF", swap_network=False):

        if molecule.spin != 0:
            raise NotImplementedError("pUCCD is implemented onmly for closed-shell system.")

        self.molecule = molecule
        self.n_spatial_orbitals = molecule.n_active_mos
        self.n_electrons = molecule.n_active_electrons

        # Set total number of parameters
        self.n_occupied = int(self.n_electrons / 2)
        self.n_virtual = self.n_spatial_orbitals - self.n_occupied
        self.n_var_params = self.n_occupied * self.n_virtual

        # Supported reference state initialization.
        self.supported_reference_state = {"HF", "zero"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "random"}

        # Default initial parameters for initialization.
        self.var_params_default = "zeros"
        self.reference_state = reference_state

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """TDB.
        """
        pass

    def prepare_reference_state(self):
        """TBD.
        """

        if self.reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.reference_state == "HF":
            vector = [1 if i < self.n_electrons // 2 else 0 for i in range(self.n_spatial_orbitals)]
            print(vector)
            return vector_to_circuit(vector)
        elif self.reference_state == "zero":
            return Circuit()

    def build_circuit(self, var_params=None):
        """TBD.
        """
        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit_operator() if self.spin == 0 else self._get_openshell_qubit_operator()

    def update_var_params(self, var_params):
        """TBD.
        """
        pass

    def _get_boson_excitations(self):
        """TBD"""

        boson_op = uccsd_singlet_generator(self.var_params, self.n_spinorbitals, self.n_electrons)
