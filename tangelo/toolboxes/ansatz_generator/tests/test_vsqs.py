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

import numpy as np
from openfermion import load_operator

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings import jordan_wigner, symmetry_conserving_bravyi_kitaev
from tangelo.toolboxes.ansatz_generator import VSQS
from tangelo.linq import Simulator
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class VSQSTest(unittest.TestCase):

    def test_vsqs_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (list, numpy array).
        """

        vsqs_ansatz = VSQS(mol_H2_sto3g)

        two_ones = np.ones((2,))

        vsqs_ansatz.set_var_params([1., 1.])
        np.testing.assert_array_almost_equal(vsqs_ansatz.var_params, two_ones, decimal=6)

        vsqs_ansatz.set_var_params(np.array([1., 1.]))
        np.testing.assert_array_almost_equal(vsqs_ansatz.var_params, two_ones, decimal=6)

    def test_vsqs_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        vsqs_ansatz = VSQS(mol_H2_sto3g)

        self.assertRaises(ValueError, vsqs_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_vsqs_H2(self):
        """Verify closed-shell VSQS functionalities for H2."""

        # Build circuit
        vsqs_ansatz = VSQS(mol_H2_sto3g, intervals=3, time=3)
        vsqs_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        vsqs_ansatz.update_var_params([0.66666667, 0.9698286, 0.21132472, 0.6465473])
        energy = sim.get_expectation_value(qubit_hamiltonian, vsqs_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701255155757, delta=1e-6)

    def test_vsqs_H4_doublecation(self):
        """Verify closed-shell VSQS functionalities for H4 2+ by using saved qubit hamiltonian and initial hamiltonian"""

        var_params = [-2.53957674, 0.72683888, 1.08799500, 0.49836183,
                      -0.23020698, 0.93278630, 0.50591026, 0.50486903]

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = load_operator("mol_H4_doublecation_minao_qubitham_jw_b.data", data_directory=pwd_this_test+"/data", plain_text=True)
        initial_hamiltonian = load_operator("mol_H4_doublecation_minao_init_qubitham_jw_b.data", data_directory=pwd_this_test+"/data", plain_text=True)
        reference_state = get_reference_circuit(8, 2, "jw", up_then_down=True, spin=0)

        # Build circuit
        vsqs_ansatz = VSQS(qubit_hamiltonian=qubit_hamiltonian, h_init=initial_hamiltonian, reference_state=reference_state,
                           intervals=5, time=5, trotter_order=2)
        vsqs_ansatz.build_circuit()

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        vsqs_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, vsqs_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.85425, delta=1e-4)

    def test_vsqs_H2_with_h_nav(self):
        """Verify closed-shell VSQS functionalities for H2 with navigator hamiltonian"""
        navigator_hamiltonian = (QubitOperator('X0 Y1', 0.03632537110234512) + QubitOperator('Y0 X1', 0.03632537110234512)
                              + QubitOperator('Y0', 2.e-5) + QubitOperator('Y1', 2.e-5))

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = symmetry_conserving_bravyi_kitaev(mol_H2_sto3g.fermionic_hamiltonian, 4, 2, False, 0)

        # Build circuit
        vsqs_ansatz = VSQS(mol_H2_sto3g, intervals=2, time=1, mapping='scbk', up_then_down=True, trotter_order=2,
                           h_nav=navigator_hamiltonian)
        vsqs_ansatz.build_circuit()

        # Assert energy returned is as expected for given parameters
        sim = Simulator()
        vsqs_ansatz.update_var_params([0.50000001, -0.02494214, 3.15398767])
        energy = sim.get_expectation_value(qubit_hamiltonian, vsqs_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372701255155757, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
