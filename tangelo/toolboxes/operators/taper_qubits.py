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

"""Module defining a class to store objects related to qubit tapering, i.e.
resource reduction through symmetries.
"""

import numpy as np

from tangelo.toolboxes.operators import MultiformOperator
from tangelo.toolboxes.operators.z2_tapering import get_clifford_operators, get_unitary, get_eigenvalues, get_z2_taper_function


class QubitTapering:
    """Class keeping track of symmetries and operation to taper qubits in qubit
    operators. The taper function is kept as an attribute to taper other Pauli
    word generators (Hamiltonian dressing, ansatz construction, etc). Z2
    tapering is implemented, but other tapering methods could be added in the
    core of this class.

    Attributes:
        initial_op (MultiformOperator): Qubit operator to be analyzed for
            symmetries.
        initial_n_qubits (int): Number of qubits before tapering.
        z2_taper (func): Function handle for tapering a MultiformOperator.
        z2_tapered_op (MultiformOperator): Tapered operator with z2 symmetries.
        z2_properties (dict): Relevant quantities used to define the z2 taper
            function. Needed to back track a z2 tapered operator to the full
            operator.
    """

    def __init__(self, qubit_operator, n_qubits, n_electrons, spin=0, mapping="JW", up_then_down=False):
        """Class keeping track of symmetries and operation to taper qubits in
        qubit operators.

        Args:
            qubit_operator (QubitOperator): Self-explanatory.
            n_qubits (int): Number of qubits the initial operator acts on.
            n_electrons (int): Number of electrons.
            mapping (str): Qubit mapping.
            up_then_down (bool): Whether or not spin ordering is all up then
                all down.
        """

        if mapping.upper() not in {"JW", "BK", "JKMN"}:
            raise NotImplementedError(f"Qubit mapping {mapping} not supported. Tapering supports JW, BK and JKMN qubit encoding.")

        self.initial_op = MultiformOperator.from_qubitop(qubit_operator, n_qubits)
        self.initial_n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.spin = spin
        self.mapping = mapping
        self.up_then_down = up_then_down

        self.z2_taper = None
        self.z2_tapered_op = None
        self.z2_properties = dict()

        self._compute_z2_symmetries()

    def z2_tapering(self, qubit_operator, n_qubits=None):
        """Function to taper a qubit operator from symmetries found in
        self.initial_op.

        Args:
            qubit_operator (QubitOperator): Self-explanatory.
            n_qubits (int): Self-explanatory.

        Returns:
            QubitOperator: The tapered qubit operator.
        """

        hybrid_op = MultiformOperator.from_qubitop(qubit_operator, n_qubits)
        z2_tapered_op = self.z2_taper(hybrid_op)

        return z2_tapered_op.qubitoperator

    def _compute_z2_symmetries(self):
        """Computes the underlying z2 symmetries in elf.initial_op. The
        procedure is described in
        Tapering off qubits to simulate fermionic Hamiltonians
        Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme.
        arXiv:1701.08213
        """

        kernel = self.initial_op.get_kernel()
        cliffords, q_indices = get_clifford_operators(kernel)
        n_symmetries = len(q_indices)
        unitary = get_unitary(cliffords)

        kernel_operator = MultiformOperator.from_binaryop(kernel, factors=np.ones(kernel.shape[0]))
        eigenvalues = get_eigenvalues(kernel_operator.binary, self.initial_n_qubits, self.n_electrons, self.spin, self.mapping, self.up_then_down)

        self.z2_taper = get_z2_taper_function(unitary, kernel_operator, q_indices, self.initial_n_qubits, n_symmetries, eigenvalues)
        self.z2_tapered_op = self.z2_taper(self.initial_op)
        self.z2_properties = {"n_symmetries": n_symmetries, "eigenvalues": eigenvalues, "unitary": unitary}
