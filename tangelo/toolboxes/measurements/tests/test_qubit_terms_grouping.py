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

import unittest
import os

from openfermion import load_operator
from openfermion.ops import QubitOperator

from tangelo.linq import translator, Simulator, Circuit
from tangelo.helpers import measurement_basis_gates
from tangelo.toolboxes.measurements import group_qwc, exp_value_from_measurement_bases, \
    check_bases_commute_qwc, map_measurements_qwc

path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'


class TermsGroupingTest(unittest.TestCase):

    def test_qubitwise_commutativity(self):
        """ Tests for check_bases_commute_qwc function. """

        I0 = ()
        Z1 = ((1, 'Z'),)
        X2 = ((2, 'X'),)
        X0Z1 = ((0, 'X'), (1, 'Z'))
        Z0Z1 = ((0, 'Z'), (1, 'Z'))

        self.assertTrue(check_bases_commute_qwc(I0, Z1))
        self.assertTrue(check_bases_commute_qwc(X2, Z1))
        self.assertTrue(check_bases_commute_qwc(Z1, X0Z1))
        self.assertTrue(check_bases_commute_qwc(Z1, Z0Z1))
        self.assertFalse(check_bases_commute_qwc(Z0Z1, X0Z1))

    def test_mmap_qwc_H2(self):
        """ From a partitioned Hamiltonian, build reverse dictionary of measurements """

        qb_ham = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)
        partitioned_ham = group_qwc(qb_ham)
        res = map_measurements_qwc(partitioned_ham)

        for k, v in res.items():
            if k in {((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')), ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')),
                     ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')), ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X'))}:
                self.assertTrue(v[0] == k)
            else:
                self.assertTrue(v[0] == ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z')))

    def test_mmap_qwc_H10frag(self):
        """ From a partitioned Hamiltonian, build reverse dictionary of measurements """

        partitioned_ham = {
            ((0, 'X'), (1, 'X')): 0.25 * QubitOperator('X0 X1'),
            ((0, 'X'), (1, 'Z')): 0.25 * (QubitOperator('X0') + QubitOperator('X0 Z1') + QubitOperator('Z1')),
            ((0, 'Y'), (1, 'Y')): -0.25 * QubitOperator('Y0 Y1'),
            ((0, 'Z'), (1, 'X')): 0.25 * (QubitOperator('Z0') + QubitOperator('Z0 X1') + QubitOperator('X1')),
            ((0, 'Z'), (1, 'Z')): 0.25 * QubitOperator('Z0 Z1')
        }
        res = map_measurements_qwc(partitioned_ham)

        ref_H10frag = {((0, 'X'), (1, 'X')): [((0, 'X'), (1, 'X'))],
                       ((0, 'X'),): [((0, 'X'), (1, 'X')), ((0, 'X'), (1, 'Z'))],
                       ((0, 'X'), (1, 'Z')): [((0, 'X'), (1, 'Z'))],
                       ((1, 'Z'),): [((0, 'X'), (1, 'Z')), ((0, 'Z'), (1, 'Z'))],
                       ((0, 'Y'), (1, 'Y')): [((0, 'Y'), (1, 'Y'))],
                       ((0, 'Z'),): [((0, 'Z'), (1, 'X')), ((0, 'Z'), (1, 'Z'))],
                       ((0, 'Z'), (1, 'X')): [((0, 'Z'), (1, 'X'))],
                       ((1, 'X'),): [((0, 'X'), (1, 'X')), ((0, 'Z'), (1, 'X'))],
                       ((0, 'Z'), (1, 'Z')): [((0, 'Z'), (1, 'Z'))]}

        self.assertDictEqual(res, ref_H10frag)

    def test_qubitwise_commutativity_of_H2(self):
        """ The JW Pauli hamiltonian of H2 at optimal geometry is a 15-term operator. Using qubitwise-commutativity,
        it is possible to get the expectation value of all 15 terms by only performing measurements in 5 distinct
        measurement bases. This test first verifies the 5 measurement bases have been identified, and then derives
        the expectation value for the qubit Hamiltonian.
        """

        # Load qubit Hamiltonian
        qb_ham = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

        # Group Hamiltonian terms using qubitwise commutativity
        grouped_ops = group_qwc(qb_ham, seed=0)

        # Load an optimized quantum circuit (UCCSD) to compute something meaningful in this test
        with open(f"{path_data}/H2_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Only simulate and measure the wavefunction in the required bases (simulator or QPU), store in dict.
        histograms = dict()
        sim = Simulator()
        for basis, sub_op in grouped_ops.items():
            full_circuit = abs_circ + Circuit(measurement_basis_gates(basis))
            histograms[basis], _ = sim.simulate(full_circuit)

        # Reconstruct exp value of initial input operator using the histograms corresponding to the suboperators
        exp_value = exp_value_from_measurement_bases(grouped_ops, histograms)
        self.assertAlmostEqual(exp_value, sim.get_expectation_value(qb_ham, abs_circ), places=8)

    def test_qubitwise_commutativity_of_H4(self):
        """ Estimating the energy of a rectangle configuration of H4, resulting in a 185-term qubit operator.
        Uses qubitwise-commutativity to identify the number of measurement bases needed (~60) to compute the
        expectation value of the full operator using a pre-loaded UCCSD H4 circuit.
        """

        # Load qubit Hamiltonian
        qb_ham = load_operator("mol_H4_qubitham.data", data_directory=path_data, plain_text=True)

        # Group Hamiltonian terms using qubitwise commutativity
        grouped_ops = group_qwc(qb_ham, seed=0)

        # Load an optimized quantum circuit (UCCSD) to compute something meaningful in this test
        with open(f"{path_data}/H4_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Only simulate and measure the wavefunction in the required bases (simulator or QPU), store in dict.
        histograms = dict()
        sim = Simulator()
        for basis, sub_op in grouped_ops.items():
            full_circuit = abs_circ + Circuit(measurement_basis_gates(basis))
            histograms[basis], _ = sim.simulate(full_circuit)

        # Reconstruct exp value of initial input operator using the histograms corresponding to the suboperators
        exp_value = exp_value_from_measurement_bases(grouped_ops, histograms)
        self.assertAlmostEqual(exp_value, sim.get_expectation_value(qb_ham, abs_circ), places=8)


if __name__ == "__main__":
    unittest.main()
