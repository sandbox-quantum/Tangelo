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

"""Docs.
"""

import numpy as np

from tangelo.toolboxes.operators.hybridoperator import HybridOperator
from tangelo.toolboxes.operators.z2_tapering import get_clifford_operators, get_unitary, get_eigenvalues, get_z2_taper_function

class QubitTapering:

    def __init__(self, qubit_operator, n_qubits, n_electrons, mapping="JW", up_then_down=False):
        """Docs."""

        self.initial_op = HybridOperator.from_qubitop(qubit_operator, n_qubits)
        self.initial_n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.mapping = mapping
        self.up_then_down = up_then_down

        self.z2_taper = None
        self.z2_tapered_op = None
        self.z2_properties = dict()

        self.compute_z2_symmetries()

    def z2_tapering(self, qubit_operator, n_qubits=None):
        """Docs."""

        hybrid_op = HybridOperator.from_qubitop(qubit_operator, n_qubits)
        z2_tapered_op = self.z2_taper(hybrid_op)

        return z2_tapered_op.qubitoperator

    def compute_z2_symmetries(self):
        """Docs."""

        kernel = self.initial_op.get_kernel()

        list_of_cliffords, q_indices = get_clifford_operators(kernel)
        n_symmetries = len(q_indices)
        unitary = get_unitary(list_of_cliffords)

        kernel_operator = HybridOperator.from_binaryop(kernel, factors=np.ones(kernel.shape[0]))
        eigenvalues = get_eigenvalues(kernel_operator.binary, self.initial_n_qubits, self.n_electrons, self.mapping, self.up_then_down)

        self.z2_taper = get_z2_taper_function(unitary, kernel_operator, q_indices, self.initial_n_qubits, n_symmetries, eigenvalues)
        self.z2_tapered_op = self.z2_taper(self.initial_op)
        self.z2_properties = {"n_symmetries": n_symmetries, "eigenvalues": eigenvalues, "unitary": unitary}
