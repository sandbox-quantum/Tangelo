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

from tangelo.toolboxes.measurements import ClassicalShadow
from tangelo.linq.circuit import Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates, pauli_string_to_of


class DerandomizedClassicalShadow(ClassicalShadow):
    """Classical shadows using derandomized single Pauli measurements, as
    defined in H.-Y. Huang, R. Kueng, and J. Preskill, ArXiv:2103.07510
    [Quant-Ph] (2021).
    """

    def build(self, n_shots, qu_op, eta=0.9):
        """Derandomized sampling of single Pauli words.

        Args:
            n_shots (int): Total number of measurements.
            qu_op (QubitOperator): Relevant QubitOperator.
            eta (float): Empirical parameter for the cost function. Default is
                0.9.

        Returns:
            list of str: Measurements generated for a derandomized procedure.
        """

        # Getting the weights, proportional to the coefficient of each Pauli
        # words. Some variables are defined to normalize the weights and track
        # the amount of measurements already defined by the algorithm.
        observables, weights = zip(*[(obs, abs(w)) for obs, w in qu_op.terms.items() if obs])

        n_observables = len(observables)
        n_measurements_per_observable = floor(n_shots / n_observables)

        norm_factor = n_observables / sum(weights)
        weights = [w * norm_factor for w in weights]

        weighted_n_measurements_per_observable = [floor(weights[i]*n_measurements_per_observable) for i in range(n_observables)]

        n_measurements_so_far = [0] * n_observables

        # Output variable (containing all chosen basis).
        measurement_procedure = list()

        # Distribution of n_measurements_per_observable * n_observables shots.
        for _ in range(n_measurements_per_observable * n_observables):

            # A single round of parallel measurements over all qubits.
            n_matches_needed_round = [len(p) for p in observables]

            single_round_measurement = [None] * self.n_qubits

            # Optimizing which Pauli basis to use for each qubit according to
            # self._cost_function.
            for i_qubit in range(self.n_qubits):
                cost_of_outcomes = {"X": 0, "Y": 0, "Z": 0}

                # Computing the cost function with all the possibilities.
                for dice_roll_pauli in ["Z", "X", "Y"]:
                    # Assume the dice rollout to be dice_roll_pauli.
                    try_matches_needed_round = n_matches_needed_round.copy()
                    for i_obs, obs in enumerate(observables):
                        try_matches_needed_round[i_obs] += _get_match_up(i_qubit, dice_roll_pauli, obs, self.n_qubits)

                    cost_of_outcomes[dice_roll_pauli] = self._cost_function(n_measurements_so_far,
                                                                            try_matches_needed_round,
                                                                            weights,
                                                                            weighted_n_measurements_per_observable,
                                                                            eta)

                # Determining the single Pauli gate to use.
                for dice_roll_pauli in ["Z", "X", "Y"]:
                    if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                        continue

                    # The best dice roll outcome will be chosen here.
                    single_round_measurement[i_qubit] = dice_roll_pauli

                    for i_obs, obs in enumerate(observables):
                        n_matches_needed_round[i_obs] += _get_match_up(i_qubit, dice_roll_pauli, obs, self.n_qubits)
                    break

            measurement_procedure.append(single_round_measurement)

            # Incrementing the number of measurements so far if there is no more
            # matches to make this round.
            n_measurements_so_far = [n[0] + 1 if n[1] == 0 else n[0] for n in zip(n_measurements_so_far, n_matches_needed_round)]

            # Incrementing success variable if number of measurements so far is
            # bigger than the weighted number of measurements per observable.
            success = sum([1 for i in range(n_observables) if n_measurements_so_far[i] >= weighted_n_measurements_per_observable[i]])

            if success == n_observables:
                break

        measurement_procedure = ["".join(m) for m in measurement_procedure]

        # Fill "missing" shots with a set of random (already chosen) basis.
        measurement_procedure += random.choices(measurement_procedure, k=n_shots-len(measurement_procedure))

        self.unitaries += measurement_procedure
        return measurement_procedure

    def get_basis_circuits(self, only_unique=False):
        """Outputs a list of circuits corresponding to the chosen single-Pauli
        unitaries.

        Args:
            only_unique (bool): Considering only unique unitaries.

        Returns:
            list of Circuit or tuple: All basis circuits or a tuple of unique
                circuits, string representation and numbers of occurence.
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
            unique_basis_circuits = [(basis_circuits[i], u, self.unitaries.count(u)) for i, u in enumerate(unitaries_to_convert)]
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

    def _cost_function(self, n_measurements_so_far, n_matches_needed, weights, weighted_num_measurements_per_observable, eta=0.9):
        """Cost function for derandomized Pauli measurements, according to
        equation (6) in the cited paper.

        Args:
            n_measurements_so_far (list of int): Number of measurements decided
                per terms.
            n_matches_needed (list of int): Number of matches.
            weights (list of float): Coefficient (absolute) of each each term.
            weighted_num_measurements_per_observable (list of float): Weighted
                number of measurements per term.
            eta (float): Empirical parameter, default set to 0.9.

        Returns:
            float: Output the cost considering the arguments.
        """

        # Computing the cost. Variables names are consistent of what is found in
        # the publication.
        nu = 1 - exp(-eta / 2)

        cost = 0.
        for (n_measure, n_match, weight, weight_per_obs) in zip(n_measurements_so_far,
                                                                n_matches_needed,
                                                                weights,
                                                                weighted_num_measurements_per_observable):

            if n_measure >= weight_per_obs:
                continue

            v = eta / 2 * n_measure
            if self.n_qubits >= n_match:
                v -= log(1 - nu / (3**n_match))

            cost += exp(-v / weight)

        return cost


def _get_match_up(lookup_qubit, dice_roll_pauli, observable, n_qubits):
    """Helper function to output 0, -1 or a large number depending on the
    index provided and the Pauli gate in a single observable.

    Args:
        lookup_qubit (int): Qubit index.
        dice_roll_pauli (str): Z, X or Y.
        observable (tuple): Single term in the form ((0, "Y"), (1, "Z"),
            (2, "X"), ...).
        n_qubits (int): Number of qubits.

    Returns:
        int: 0, -1, or a large int used in computing cost.
    """

    large_number = 100 * (n_qubits + 10)

    for i_qubit, pauli in observable:
        if lookup_qubit == i_qubit:
            return -1 if pauli == dice_roll_pauli else large_number
    return 0
