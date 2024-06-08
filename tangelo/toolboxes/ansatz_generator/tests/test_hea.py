# Copyright SandboxAQ 2021-2024.
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
import numpy as np
import os
from openfermion import load_operator

from tangelo.molecule_library import mol_H2_sto3g, mol_H4_doublecation_minao
from tangelo.toolboxes.qubit_mappings import jordan_wigner
from tangelo.toolboxes.ansatz_generator.hea import HEA
from tangelo.toolboxes.operators import count_qubits
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit

from tangelo.linq import Circuit, Gate, get_backend

# Initiate simulator
sim = get_backend()

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))


class HEATest(unittest.TestCase):

    def test_hea_set_var_params(self):
        """Verify behavior of set_var_params for different inputs (keyword,
        list, numpy array).
        """

        hea_ansatz = HEA(molecule=mol_H2_sto3g)

        hea_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.ones(4 * 3 * 3), decimal=6)

        hea_ansatz.set_var_params(np.ones(4 * 3 * 3))
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.ones(4 * 3 * 3), decimal=6)

        hea_ansatz.set_var_params("zeros")
        np.testing.assert_array_almost_equal(hea_ansatz.var_params, np.zeros(4 * 3 * 3), decimal=6)

    def test_hea_incorrect_number_var_params(self):
        """Return an error if user provide incorrect number of variational
        parameters.
        """

        hea_ansatz = HEA(molecule=mol_H2_sto3g)

        self.assertRaises(ValueError, hea_ansatz.set_var_params, np.ones(4 * 3 * 3 + 1))

    def test_hea_H2(self):
        """Verify closed-shell HEA functionalities for H2."""

        # Build circuit
        hea_ansatz = HEA(molecule=mol_H2_sto3g)
        hea_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)

        params = [ 1.96262489e+00, -9.83505909e-01, -1.92659544e+00,  3.68638855e+00,
                   4.71410736e+00,  4.78247991e+00,  4.71258582e+00, -4.79077006e+00,
                   4.60613188e+00, -3.14130503e+00,  4.71232383e+00,  1.35715841e+00,
                   4.30998410e+00, -3.00415626e+00, -1.06784872e+00, -2.05119893e+00,
                   6.44114344e+00, -1.56358255e+00, -6.28254779e+00,  3.14118427e+00,
                  -3.10505551e+00,  3.15123780e+00, -3.64794717e+00,  1.09127829e+00,
                   4.67093656e-01, -1.19912860e-01,  3.99351728e-03, -1.57104046e+00,
                   1.56811666e+00, -3.14050540e+00,  4.71181097e+00,  1.57036595e+00,
                  -2.16414405e+00,  3.40295404e+00, -2.87986715e+00, -1.69054279e+00]

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372661564779496, delta=1e-6)

    def test_hea_H4_doublecation(self):
        """Verify closed-shell HEA functionalities for H4 2+."""

        var_params = [-2.34142720e-04,  6.28472420e+00,  4.67668267e+00, -3.14063369e+00,
                      -1.53697174e+00, -6.22546556e+00, -3.11351342e+00,  3.14158366e+00,
                      -1.57812617e+00,  3.14254101e+00, -6.29656957e+00, -8.93646210e+00,
                       7.84465163e-04, -1.32569792e-05,  1.90710480e+00,  6.44924149e+00,
                      -4.69797158e+00,  3.47688734e+00,  8.24677008e-04,  1.89006312e-04,
                      -1.57734737e+00, -3.63191375e+00, -3.15706332e+00, -1.73625091e+00]

        # Build circuit with Ry rotationas instead of RZ*RX*RZ rotations
        hea_ansatz = HEA(molecule=mol_H4_doublecation_minao, rot_type="real")
        hea_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        qubit_hamiltonian = load_operator("mol_H4_doublecation_minao_qubitham_jw.data", data_directory=pwd_this_test+"/data", plain_text=True)

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -0.795317, delta=1e-4)

    def test_hea_circuit_variational_reference_state(self):
        """ Verify variational gate parameters are frozen in a reference state circuit"""

        # Create simple reference circuit:
        ref_circuit = Circuit([
            Gate('RY', 0, parameter=1.0, is_variational=True),
            Gate('RY', 1, parameter=1.0, is_variational=True),
            Gate('CX', 0, 1),
        ])

        # Create HEA ansatz with reference circuit
        hea_ansatz = HEA(n_qubits=2, reference_state=ref_circuit)
        hea_circ = hea_ansatz.build_circuit()

        # Ensure gates are correctly prepended to circuit
        self.assertEqual(hea_circ.width, 2)

        # Ensure reference circuit gates were correctly converted
        # to non-variational gates with the same name
        hea_circ_gate_names = [ gate.name for gate in hea_circ ]
        ref_circ_gate_names = [ gate.name for gate in ref_circuit ]
        hea_circ_gate_variationals = [ gate.is_variational for gate in hea_circ ]

        self.assertListEqual(ref_circ_gate_names, hea_circ_gate_names[:ref_circuit.size])
        self.assertTrue(not any(hea_circ_gate_variationals[:ref_circuit.size]))

    def test_hea_circuit_reference_state_H2(self):
        """ Verify construction of H2 ansatz works using a circuit reference state."""
        sim = get_backend()

        # Build qubit hamiltonian
        qubit_hamiltonian = jordan_wigner(mol_H2_sto3g.fermionic_hamiltonian)
        n_qubits = count_qubits(qubit_hamiltonian)

        # Construct reference circuit by hand
        ref_h2_circuit = get_reference_circuit(
                n_spinorbitals=n_qubits,
                n_electrons=mol_H2_sto3g.n_electrons,
                mapping='jw',
                up_then_down=False,
                spin=mol_H2_sto3g.spin)

        # Build ansatz circuit
        hea_ansatz = HEA(n_qubits=n_qubits, reference_state=ref_h2_circuit)
        hea_ansatz.build_circuit()

        params = [ 1.96262489e+00, -9.83505909e-01, -1.92659544e+00,  3.68638855e+00,
                4.71410736e+00,  4.78247991e+00,  4.71258582e+00, -4.79077006e+00,
                4.60613188e+00, -3.14130503e+00,  4.71232383e+00,  1.35715841e+00,
                4.30998410e+00, -3.00415626e+00, -1.06784872e+00, -2.05119893e+00,
                6.44114344e+00, -1.56358255e+00, -6.28254779e+00,  3.14118427e+00,
                -3.10505551e+00,  3.15123780e+00, -3.64794717e+00,  1.09127829e+00,
                4.67093656e-01, -1.19912860e-01,  3.99351728e-03, -1.57104046e+00,
                1.56811666e+00, -3.14050540e+00,  4.71181097e+00,  1.57036595e+00,
                -2.16414405e+00,  3.40295404e+00, -2.87986715e+00, -1.69054279e+00]

        # Assert energy returned is as expected for given parameters
        hea_ansatz.update_var_params(params)
        energy = sim.get_expectation_value(qubit_hamiltonian, hea_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372661564779496, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
