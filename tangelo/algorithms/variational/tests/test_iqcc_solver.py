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

"""Unit tests for the closed-shell and restricted open-shell iQCC-VQE Solver. """

import unittest

from tangelo.algorithms.variational import iQCCsolver
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g,\
                                     mol_H4_doublecation_minao


class iQCCsolver_test(unittest.TestCase):
    """Unit tests for the iQCCsolver class. Examples for both closed-shell
    and restricted open-shell iQCC are provided via H4, H4+, and H4+2.
    """

    @staticmethod
    def test_build_success():
        """Test instantation of iQCC solver with user-defined input."""

        iqcc_options = {"molecule": mol_H2_sto3g,
                        "qubit_mapping": "scbk",
                        "up_then_down": True,
                        "deqcc_thresh": 1e-5,
                        "max_iqcc_iter": 25,
                        "max_iqcc_retries": 10,
                        "compress_qubit_ham": True,
                        "compress_eps": 1e-4}

        iqcc_solver = iQCCsolver(iqcc_options)
        iqcc_solver.build()

    def test_build_fail(self):
        """Test that instantation of iQCC solver fails without input of a molecule."""

        iqcc_options = {"max_iqcc_iter": 15}
        self.assertRaises(ValueError, iQCCsolver, iqcc_options)

    def test_iqcc_h4(self):
        """Test the final energy for a complete iQCC solver loop for H4 using the
        maximum number of generators and compressing the qubit Hamiltonian"""

        ansatz_options = {"max_qcc_gens": None}

        iqcc_options = {"molecule": mol_H4_sto3g,
                        "qubit_mapping": "scbk",
                        "up_then_down": True,
                        "ansatz_options": ansatz_options,
                        "deqcc_thresh": 1e-5,
                        "max_iqcc_iter": 50,
                        "compress_qubit_ham": True,
                        "compress_eps": 1e-4}

        iqcc_solver = iQCCsolver(iqcc_options)
        iqcc_solver.build()
        iqcc_energy = iqcc_solver.simulate()

        self.assertAlmostEqual(iqcc_energy, -1.977505, places=4)

    def test_iqcc_h4_cation(self):
        """Test the final energy for a complete iQCC solver loop for H4+"""

        ansatz_options = {"max_qcc_gens": None}

        iqcc_options = {"molecule": mol_H4_cation_sto3g,
                        "qubit_mapping": "scbk",
                        "up_then_down": True,
                        "ansatz_options": ansatz_options,
                        "deqcc_thresh": 1e-5,
                        "max_iqcc_iter": 50}

        iqcc_solver = iQCCsolver(iqcc_options)
        iqcc_solver.build()
        iqcc_energy = iqcc_solver.simulate()

        self.assertAlmostEqual(iqcc_energy, -1.638526, places=4)

    def test_iqcc_h4_double_cation(self):
        """Test the energy for a single iQCC iteration for H4+2"""

        ansatz_options = {"max_qcc_gens": None}

        iqcc_options = {"molecule": mol_H4_doublecation_minao,
                        "qubit_mapping": "scbk",
                        "up_then_down": True,
                        "ansatz_options": ansatz_options,
                        "deqcc_thresh": 1e-5,
                        "max_iqcc_iter": 1}

        iqcc_solver = iQCCsolver(iqcc_options)
        iqcc_solver.build()
        iqcc_energy = iqcc_solver.simulate()

        self.assertAlmostEqual(iqcc_energy, -0.854647, places=4)


if __name__ == "__main__":
    unittest.main()
