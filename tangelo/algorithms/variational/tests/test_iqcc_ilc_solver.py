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

"""Unit tests for the closed-shell and restricted open-shell iQCC-ILC solver."""

import unittest

from tangelo.algorithms.variational import iQCC_ILC_solver
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g_symm, mol_H4_cation_sto3g


class iQCC_ILC_solver_test(unittest.TestCase):
    """Unit tests for the iQCC_ILC_solver class. Examples for both closed-shell
    and restricted open-shell iQCC-ILC are provided via H4 and H4+.
    """

    @staticmethod
    def test_build_success():
        """Test instantation of the iQCC-ILC solver with user-defined input."""

        iqcc_ilc_options = {"molecule": mol_H2_sto3g,
                            "qubit_mapping": "scbk",
                            "up_then_down": True,
                            "max_ilc_iter": 25,
                            "compress_qubit_ham": True,
                            "compress_eps": 1e-4}

        iqcc_ilc = iQCC_ILC_solver(iqcc_ilc_options)
        iqcc_ilc.build()

    def test_build_fail(self):
        """Test that instantation of the iQCC-ILC solver fails without input of a molecule."""

        iqcc_ilc_options = {"max_ilc_iter": 15}
        self.assertRaises(ValueError, iQCC_ILC_solver, iqcc_ilc_options)

    def test_iqcc_ilc_h4(self):
        """Test the energy after 1 iteration for H4."""

        ilc_ansatz_options = {"max_ilc_gens": None}
        qcc_ansatz_options = {"max_qcc_gens": None}

        iqcc_ilc_options = {"molecule": mol_H4_sto3g_symm,
                            "qubit_mapping": "scbk",
                            "up_then_down": True,
                            "ilc_ansatz_options": ilc_ansatz_options,
                            "qcc_ansatz_options": qcc_ansatz_options,
                            "max_ilc_iter": 1}

        iqcc_ilc_solver = iQCC_ILC_solver(iqcc_ilc_options)
        iqcc_ilc_solver.build()
        iqcc_ilc_energy = iqcc_ilc_solver.simulate()

        self.assertAlmostEqual(iqcc_ilc_energy, -1.976817, places=4)

    def test_iqcc_ilc_h4_cation(self):
        """Test the energy after 2 iterations for H4+ using the maximum
        number of generators and compressing the qubit Hamiltonian."""

        ilc_ansatz_options = {"max_ilc_gens": None}
        qcc_ansatz_options = {"max_qcc_gens": None}

        iqcc_ilc_options = {"molecule": mol_H4_cation_sto3g,
                            "qubit_mapping": "scbk",
                            "up_then_down": True,
                            "ilc_ansatz_options": ilc_ansatz_options,
                            "qcc_ansatz_options": qcc_ansatz_options,
                            "max_ilc_iter": 2,
                            "compress_qubit_ham": True,
                            "compress_eps": 1e-4}

        iqcc_ilc_solver = iQCC_ILC_solver(iqcc_ilc_options)
        iqcc_ilc_solver.build()
        iqcc_ilc_energy = iqcc_ilc_solver.simulate()

        self.assertAlmostEqual(iqcc_ilc_energy, -1.638197, places=4)


if __name__ == "__main__":
    unittest.main()
