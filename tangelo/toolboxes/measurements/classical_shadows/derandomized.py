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

"""This file provides an API enabling the use of derandomized classical shadows.
This algorithm is described in H.-Y. Huang, R. Kueng, and J. Preskill,
ArXiv:2103.07510 [Quant-Ph] (2021).
"""

import math
import random

import numpy as np

from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of


class DerandomizedClassicalShadow(ClassicalShadow):
    r"""Classical shadows using randomized single Pauli measurements, as defined
    in H.Y. Huang, R. Kueng, and J. Preskill, Nature Physics 16, 1050 (2020). In
    short, the channel is inversed to geet the state with the formula
    \hat{\rho} = \bigotimes_{j=1}^n \left( 3U_j^{\dagger} |b_j\rangle \langle b_j| U_j - \mathbb{I} \right)
    """

    def build(self, n_shots, qu_op, eta=0.9):
        observables = [obs for obs in qu_op.terms.keys() if obs]
        weights = [abs(w) for w in qu_op.terms.values()]

        n_observables = len(observables)
        n_measurements_per_observable = round(n_shots / n_observables)

        norm_factor = n_observables / sum(weights)
        weights = [w*norm_factor for w in weights]

        # weight = [1.0] * num_observables
        #assert(False)

        weighted_num_measurements_per_observable = np.empty(n_observables, dtype=np.int64)
        for i in range(n_observables):
            weighted_num_measurements_per_observable[i] = math.floor(weights[i]*n_measurements_per_observable)

        n_measurements_so_far = np.zeros(n_observables, dtype=np.int64)  # [0] * num_observables
        measurement_procedure = []

        large_number = 100 * (self.n_qubits+10)

        for _ in range(n_measurements_per_observable * n_observables):
            # A single round of parallel measurement over "self.n_qubits" number of qubits
            num_of_matches_needed_in_this_round = [len(p) for p in observables]
            single_round_measurement = np.empty(self.n_qubits, dtype='<U1')

            for qubit_i in range(self.n_qubits):
                cost_of_outcomes = dict([("X", 0), ("Y", 0), ("Z", 0)])

                for dice_roll_pauli in ["X", "Y", "Z"]:
                    # Assume the dice rollout to be "dice_roll_pauli"
                    for i, single_observable in enumerate(observables):
                        result = self._match_up(qubit_i, dice_roll_pauli, single_observable)
                        if result == -1:
                            num_of_matches_needed_in_this_round[i] += large_number  # impossible to measure
                        if result == 1:
                            num_of_matches_needed_in_this_round[i] -= 1  # match up one Pauli X/Y/Z

                    cost_of_outcomes[dice_roll_pauli] = self._cost_function(n_measurements_so_far, num_of_matches_needed_in_this_round, weights, weighted_num_measurements_per_observable, eta)

                    # Revert the dice roll
                    for i, single_observable in enumerate(observables):
                        result = self._match_up(qubit_i, dice_roll_pauli, single_observable)
                        if result == -1:
                            num_of_matches_needed_in_this_round[i] -= large_number  # impossible to measure
                        if result == 1:
                            num_of_matches_needed_in_this_round[i] += 1  # match up one Pauli X/Y/Z

                for dice_roll_pauli in ["X", "Y", "Z"]:
                    if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                        continue
                    # The best dice roll outcome will come to this line
                    single_round_measurement[qubit_i] = dice_roll_pauli
                    for i, single_observable in enumerate(observables):
                        result = self._match_up(qubit_i, dice_roll_pauli, single_observable)
                        if result == -1:
                            num_of_matches_needed_in_this_round[i] += large_number  # impossible to measure
                        if result == 1:
                            num_of_matches_needed_in_this_round[i] -= 1  # match up one Pauli X/Y/Z
                    break

            measurement_procedure.append(single_round_measurement)

            for i, single_observable in enumerate(observables):
                if num_of_matches_needed_in_this_round[i] == 0:  # finished measuring all qubits
                    n_measurements_so_far[i] += 1

            success = 0
            for i, single_observable in enumerate(observables):
                if n_measurements_so_far[i] >= weighted_num_measurements_per_observable[i]:
                    success += 1

            if success == len(observables):
                print(success)
                break

        return measurement_procedure

    def get_basis_circuits(self, only_unique=False):
        """Output a list of circuits corresponding to the random Pauli words
        unitaries.

        Args:
            only_unique (bool): Considering only unique unitaries.

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

        # Counting each unique circuits (use for reversing to a full shadow from an experiement on hardware).
        if only_unique:
            unique_basis_circuits = [(basis_circuits[i], self.unitaries.count(u)) for i, u in enumerate(unitaries_to_convert)]
            return unique_basis_circuits
        else:
            return basis_circuits

    def estimate_state(self):
        """Doc."""
        raise NotImplementedError

    def get_term_observable(self, one_observable, coeff=1.):
        """Returns the estimated observable for a term and its coefficient.

        Args:
            term (tuple): Openfermion style of a qubit operator term.
            coeff (float): Multiplication factor for the term.
            k (int): Grouping k observations for the means of median protocol.

        Returns:
            float: Observable estimated with the shadow.
        """

        sum_product, cnt_match = 0, 0
        shadow_size = len(self)
        zero_state = 1
        one_state = -1

        # for single_measurement in shadow_size:
        for snapshot in range(shadow_size):
            not_match = 0
            product = 1

            for position, pauli_XYZ in one_observable:
                if pauli_XYZ != self.unitaries[snapshot][position]:
                    not_match = 1
                    break
                state = zero_state if self.bitstrings[snapshot][position] == "0" else one_state
                product *= state
            if not_match == 1:
                continue

            sum_product += product
            cnt_match += 1
        if cnt_match > 0:
            return sum_product/cnt_match*coeff
        else:
            return 0

    def _cost_function(self, n_measurements_so_far, n_matches_needed_in_this_round, weights, weighted_num_measurements_per_observable, eta=0.9):
        nu = 1 - math.exp(-eta / 2)

        cost = 0
        for i, zipitem in enumerate(zip(n_measurements_so_far, n_matches_needed_in_this_round)):
            measurement_so_far, matches_needed = zipitem
            if n_measurements_so_far[i] >= weighted_num_measurements_per_observable[i]:
                continue

            if self.n_qubits < matches_needed:
                v = eta / 2 * measurement_so_far
            else:
                v = eta / 2 * measurement_so_far - math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-v / weights[i])
        return cost

    def _match_up(self, qubit_i, dice_roll_pauli, single_observable):
        for pos, pauli in single_observable:
            if pos != qubit_i:
                continue
            else:
                if pauli != dice_roll_pauli:
                    return -1
                else:
                    return 1
        return 0
