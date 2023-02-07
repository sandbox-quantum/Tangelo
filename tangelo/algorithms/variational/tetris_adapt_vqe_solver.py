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

"""Module that defines the TETRIS-ADAPT-VQE algorithm framework. Differences vs
ADAPT-VQE: more than one operators acting on different qubits can be added to
the adaptive ansatz during the same ADAPT cycle. This algorithm creates denser
circuits.

Ref:
    Panagiotis G. Anastasiou, Yanzhu Chen, Nicholas J. Mayhall, Edwin Barnes,
    and Sophia E. Economou.
    TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser
    circuit ansätze
    arXiv:2209.10562 (2022)
"""

from tangelo.algorithms.variational import ADAPTSolver


class TETRISADAPTSolver(ADAPTSolver):
    """TETRIS-ADAPT-VQE class. This is an iterative algorithm that uses VQE. A
    single method is redefined from ADAPTSolver to allow the addition of many
    operators per ADAPT cycle.
    """

    def choose_operator(self, gradients, tolerance=1e-3):
        """Choose the next operator(s) to add according to the TETRIS-ADAPT-VQE
        algorithm.

        Args:
            gradients (list of float): Operator gradients (absolute values)
                corresponding to self.pool_operators.
            tolerance (float): Minimum value for gradient to be considered.

        Returns:
            list of int: Indice(s) of the operator(s) to be considered for this
                ADAPT cycle.
        """

        qubit_indices = set(range(self.ansatz.circuit.width))

        # Sorting the pool operators according to the gradients.
        sorted_op_indices = sorted(range(len(gradients)), key=lambda k: gradients[k])

        op_indices_to_add = list()
        for i in sorted_op_indices[::-1]:

            # If gradient is lower than the tolerance, all the remaining
            # operators have a lower gradient also. If there is no "available"
            # qubit anymore, no more operator can be added.
            if gradients[i] < tolerance or len(qubit_indices) == 0:
                break

            op_indices = self.pool_operators[i].qubit_indices

            # If the operator acts on a subset of "available" qubits, it can be
            # considered for this ADAPT cycle. Those qubit indices are then
            # removed from consideration for this ADAPT cycle.
            if op_indices.issubset(qubit_indices):
                qubit_indices -= op_indices
                op_indices_to_add.append(i)

        return op_indices_to_add
