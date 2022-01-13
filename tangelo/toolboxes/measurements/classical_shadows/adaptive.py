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

"""This file provides an API enabling the use of adaptive classical shadows.
This algorithm is described in C. Hadfield, ArXiv:2105.12207 [Quant-Ph] (2021).
"""

import random

import numpy as np

from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of


class AdaptiveClassicalShadow(ClassicalShadow):
    """Classical shadows using adaptive single Pauli measurements, as defined
    in C. Hadfield, ArXiv:2105.12207 [Quant-Ph] (2021).
    """

    def build(self, n_shots, qu_op):
        """Adaptive classical shadow building method to define relevant
        unitaries depending on the qubit operator.

        Args:
            n_shots (int): The number of desired measurements.
            qu_op (QubitOperator): The observable that one wishes to measure.

        Returns:
            list of str: The list of Pauli words that describes the measurement
                basis to use.
        """

        measurement_procedure = [self._choose_measurement(qu_op) for _ in range(n_shots)]

        self.unitaries = measurement_procedure
        return measurement_procedure

    def _choose_measurement(self, qu_op):
        """Algorithm 1 from the publication.

        Args:
            qu_op (QubitOperator): The operator that one wishes to maximize the
                measurement budget over.

        Returns:
            str: Pauli words for one measurement.
        """

        i_qubit_random = random.sample(range(self.n_qubits), self.n_qubits)
        inverse_map = np.argsort(i_qubit_random)

        single_measurement = [None] * self.n_qubits

        for it, i_qubit in enumerate(i_qubit_random):
            cbs = self._generate_cbs(qu_op,
                                     i_qubit_random[0:it],
                                     single_measurement[0:it],
                                     i_qubit)
            sum_cbs = sum(cbs)

            # If sum is 0., the distribution is set to be uniform.
            if sum_cbs < 1.e-7:
                single_measurement[it] = random.choice(["X", "Y", "Z"])
            else:
                random_val = random.random()

                # Depending on the cbs, the probabilities of drawing a specific
                # Pauli gate are shifted.
                if random_val < cbs[0] / sum_cbs:
                    single_measurement[it] = "X"
                elif random_val < (cbs[0] + cbs[1]) / sum_cbs:
                    single_measurement[it] = "Y"
                else:
                    single_measurement[it] = "Z"

        # Reorders according to the qubit indices 0, 1, 2, ... self.n_qubits.
        reordered_measurement = [single_measurement[inverse_map[j]] for j in range(self.n_qubits)]

        return "".join(reordered_measurement)

    def _generate_cbs(self, qu_op, prev_qubits, prev_paulis, curr_qubit):
        """Generates the cB values from which the Pauli basis is determined for
        the current qubit (curr_qubit), as shown in Algorithm 2 from the paper.

        Args:
            qu_op (QubitOperator) : The operator one wishes to get the
                expectation value of.
            prev_qubits (list) : list of previous qubits from which the
                measurement basis is already determined.
            prev_paulis (list) : the Pauli word for prev_qubits.
            curr_qubit (int) : The current qubit being examined.

        Returns:
            list of float: cB values for X, Y and Z.
        """

        cbs = [0.] * 3
        map_pauli = {"X": 0, "Y": 1, "Z": 2}

        for term, coeff in qu_op.terms.items():

            # Default conditions to compute cbs.
            same_qubit = False
            same_pauli = True

            for i_qubit, pauli in term:
                # Checks if the current qubit index has been detected in the
                # term.
                if i_qubit == curr_qubit:
                    same_qubit = True

                # Checks if the Pauli basis in the term has already been chosen
                # for this qubit.
                for prev_qubit, prev_pauli in zip(prev_qubits, prev_paulis):
                    if i_qubit == prev_qubit and pauli != prev_pauli:
                        same_pauli = False

                if same_qubit and same_pauli:
                    cbs[map_pauli[pauli]] += coeff**2

        return np.sqrt(cbs)

    def get_basis_circuits(self, only_unique=False):
        """Outputs a list of circuits corresponding to the adaptive single-Pauli
        unitaries.

        Args:
            only_unique (bool): Consider only unique unitaries.

        Returns:
            list of Circuit or tuple: All basis circuits or a tuple of unique
                circuits (first) with the numbers of occurence (last).
        """

        if not self.unitaries:
            raise ValueError(f"A set of unitaries must de defined (can be done with the build method in {self.__class__.__name__}).")

        unitaries_to_convert = self.unique_unitaries if only_unique else self.unitaries

        basis_circuits = list()
        for pauli_word in unitaries_to_convert:
            # Transformation of a unitary to quantum gates.
            pauli_of = pauli_string_to_of(pauli_word)
            basis_circuits += [Circuit(measurement_basis_gates(pauli_of), self.n_qubits)]

        # Counts each unique circuits (use for reversing to a full shadow from
        # an experiement on hardware).
        if only_unique:
            unique_basis_circuits = [(basis_circuits[i], self.unitaries.count(u)) for i, u in enumerate(unitaries_to_convert)]
            return unique_basis_circuits
        else:
            return basis_circuits

    def get_term_observable(self, term, coeff=1.):
        """Returns the estimated observable for a term and its coefficient.

        Args:
            term (tuple): Openfermion style of a qubit operator term.
            coeff (float): Multiplication factor for the term.

        Returns:
            float: Observable estimated with the shadow.
        """

        sum_product = 0
        n_match = 0

        # For every single_measurement in shadow_size.
        for snapshot in range(self.size):
            match = 1
            product = 1

            # Checks if there is a match for all Pauli gate in the term. Works
            # also with operator not on all qubits (e.g. X1 will hit Z0X1, Y0X1
            # and Z0X1).
            for i_qubit, pauli in term:
                if pauli != self.unitaries[snapshot][i_qubit]:
                    match = 0
                    break
                if self.bitstrings[snapshot][i_qubit] != "0":
                    product *= -1

            # No quantity is considered if there is no match.
            sum_product += match * product
            n_match += match

        return sum_product / n_match * coeff if n_match > 0 else 0.
