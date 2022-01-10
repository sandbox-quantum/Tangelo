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


class AdaptiveClassicalShadow(ClassicalShadow):
    """Classical shadows using adaptive single Pauli measurements, as defined
    in ...
    """
    '''Adaptive classical shadow as described in https://arxiv.org/pdf/2105.12207.pdf
       Args:
            qu_op (QubitOperator): The observable that one wishes to measure
            n_qubits (int): The number of qubits in the system
            shadow_size (int): The number of desired measurements
        Returns:
            measurement_list (array(shadow_size): The list of pauli words that describes the measurement
                                                  basis to use
    '''

    def build(self, n_shots, qu_op):

        def generate_cbs(prev_qubits, curr_qubit, blist, qu_op):
            '''Generates the cB values from which the pauli basis is determined for curr_qubit
            AKA Algorithm 2 from the paper
            Args:
                prev_qubits (list) : list of previous qubits from which the measurement basis
                                        is already determined
                curr_qubit (int) : The current qubit being examined
                blist (list) : the pauli word for prev_qubits
                qu_op (QubitOperator) : The operator one wishes to get the expectation value of
            '''
            cb = np.zeros(3, dtype=float)
            map_pauli = {'X': 0, 'Y': 1, 'Z': 2}
            for term, alphap in qu_op.terms.items():
                first_cond = False
                second_cond = True
                for pos, pauli in term:
                    if pos == curr_qubit:
                        first_cond = True
                        candidate_pauli = map_pauli[pauli]
                    for jp, qubit_jp in enumerate(prev_qubits):
                        if pos != qubit_jp:
                            continue
                        else:
                            if pauli != blist[jp]:
                                second_cond = False
                    if first_cond and second_cond:
                        cb[candidate_pauli] += abs(alphap)**2
            return np.sqrt(cb)

        def choose_measurement(n, qu_op):
            ''' Algorithm 1 from the paper
            Args:
                n (list) : list of qubits
                qu_op : The operator that one wishes to maximize the measurement budget over
                returns Pauli word for one measurement
            '''
            i_bi = random.sample(n, len(n))
            inverse_map = np.argsort(i_bi)
            single_measurement = np.empty(len(n), dtype='<U1')

            for jp, j in enumerate(i_bi):
                cbs = generate_cbs(i_bi[0:jp], j, single_measurement[0:jp], qu_op)
                den = sum(cbs)
                if den < 1.e-7:
                    B = random.choice(['X', 'Y', 'Z'])
                else:
                    dist = [cbs[0]/den, (cbs[0]+cbs[1])/den]
                    val = random.random()
                    if val < dist[0]:
                        single_measurement[jp] = 'X'
                    elif val < dist[1]:
                        single_measurement[jp] = 'Y'
                    else:
                        single_measurement[jp] = 'Z'

            return [single_measurement[inverse_map[j]] for j in range(len(n))]

        qubit_list = [i for i in range(n_qubits)]
        measurements = np.empty((shadow_size, n_qubits), dtype='<U1')
        for s in range(shadow_size):
            measurements[s] = choose_measurement(qubit_list, qu_op)

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

        Returns:
            float: Observable estimated with the shadow.
        """

        sum_product, cnt_match = 0., 0
        zero_state = 1
        one_state = -1

        # For every single_measurement in shadow_size.
        for snapshot in range(self.size):
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
            return sum_product / cnt_match * coeff
        else:
            return 0.

    def _cost_function(self, n_measurements_so_far, n_matches_needed_in_this_round, weights, weighted_num_measurements_per_observable, eta=0.9):
        """Doc"""

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
        """Doc"""

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
