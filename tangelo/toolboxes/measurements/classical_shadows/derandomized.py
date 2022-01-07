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

from math import floor, exp, log
import random

import numpy as np

from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of

# TODO: discuss about qubit_op = QubitOperator("Y1", 1.) + QubitOperator("X0", 1.) + QubitOperator("Z1", 1.)
# and how to handle that (the X).


class DerandomizedClassicalShadow(ClassicalShadow):
    r"""Classical shadows using randomized single Pauli measurements, as defined
    in H.-Y. Huang, R. Kueng, and J. Preskill, ArXiv:2103.07510 [Quant-Ph]
    (2021).
    """

    def build(self, n_shots, qu_op, eta=0.9):

        # Getting the weights, proportional to the coefficient of each Pauli
        # words. Some variables are defined to normalize the weights and track
        # the amount of measurements already defined by the algorithm.
        observables = [obs for obs in qu_op.terms.keys() if obs]
        weights = [abs(w) for w in qu_op.terms.values()]

        n_observables = len(observables)
        n_measurements_per_observable = round(n_shots / n_observables)

        norm_factor = n_observables / sum(weights)
        weights = [w*norm_factor for w in weights]

        weighted_n_measurements_per_observable = [floor(weights[i]*n_measurements_per_observable) for i in range(n_observables)]

        n_measurements_so_far = [0] * n_observables

        # Output variable (containing all basis).
        measurement_procedure = list()

        # Distribution of close to n_shots.
        for _ in range(n_measurements_per_observable * n_observables):

            # A single round of parallel measurement over "self.n_qubits" number
            # of qubits.
            n_matches_needed_round = [len(p) for p in observables]

            single_round_measurement = [None] * self.n_qubits

            # Optimizing which Pauli gate to use for each qubit according to
            # self._cost_function.
            for i_qubit in range(self.n_qubits):
                cost_of_outcomes = dict([("X", 0), ("Y", 0), ("Z", 0)])

                # Computing the cost function with all the possibilities.
                for dice_roll_pauli in ["Z", "X", "Y"]:
                    # Assume the dice rollout to be dice_roll_pauli.
                    for i_obs, obs in enumerate(observables):
                        n_matches_needed_round[i_obs] += self._match_up(i_qubit, dice_roll_pauli, obs)

                    cost_of_outcomes[dice_roll_pauli] = self._cost_function(n_measurements_so_far, n_matches_needed_round, weights, weighted_n_measurements_per_observable, eta)

                    # Revert the dice roll.
                    for i_obs, obs in enumerate(observables):
                        n_matches_needed_round[i_obs] -= self._match_up(i_qubit, dice_roll_pauli, obs)

                # Determining the single Pauli gate to use.
                for dice_roll_pauli in ["Z", "X", "Y"]:
                    if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                        continue

                    # The best dice roll outcome will come to this line.
                    single_round_measurement[i_qubit] = dice_roll_pauli

                    for i_obs, obs in enumerate(observables):
                        n_matches_needed_round[i_obs] += self._match_up(i_qubit, dice_roll_pauli, obs)
                    break

            measurement_procedure.append(single_round_measurement)

            # Incrementing the number of measurements so far if there is no more
            # matches to make this round.
            n_measurements_so_far = list(map(lambda n: n[0] + 1 if n[1] == 0 else n[0], zip(n_measurements_so_far, n_matches_needed_round)))

            # Incrementing success variable if number of measurements so far is
            # bigger than the weighted number of measurements per observable.
            success = sum([1 for i in range(n_observables) if n_measurements_so_far[i] >= weighted_n_measurements_per_observable[i]])

            if success == n_observables:
                break

        measurement_procedure = ["".join(m) for m in measurement_procedure]

        # Fill "missing" shots with a set of random (already chosen) basis.
        measurement_procedure += [random.choice(measurement_procedure) for _ in range(n_shots-len(measurement_procedure))]

        self.unitaries = measurement_procedure
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
        nu = 1 - exp(-eta / 2)

        cost = 0.
        for i_obs, (measurement_so_far, matches_needed) in enumerate(zip(n_measurements_so_far, n_matches_needed_in_this_round)):

            if n_measurements_so_far[i_obs] >= weighted_num_measurements_per_observable[i_obs]:
                continue

            v = eta / 2 * measurement_so_far
            if self.n_qubits >= matches_needed:
                v -= log(1 - nu / (3**matches_needed))

            cost += exp(-v / weights[i_obs])

        return cost

    def _match_up(self, qubit_i, dice_roll_pauli, single_observable):

        large_number = 100 * (self.n_qubits+10)

        for pos, pauli in single_observable:
            if pos != qubit_i:
                continue
            else:
                if pauli != dice_roll_pauli:
                    return large_number
                else:
                    return -1
        return 0
