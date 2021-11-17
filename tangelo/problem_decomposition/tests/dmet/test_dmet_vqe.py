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
from copy import copy

from tangelo.molecule_library import mol_H4_minao
from tangelo.problem_decomposition.dmet.dmet_problem_decomposition import Localization, DMETProblemDecomposition


class DMETVQETest(unittest.TestCase):

    def test_h4ring_vqe_uccsd(self):
        """DMET on H4 ring with fragment size one, using VQE-UCCSD."""

        opt_dmet = {"molecule": mol_H4_minao,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": ["vqe", "ccsd", "ccsd", "ccsd"],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        # Run DMET
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        energy = dmet.simulate()

        # Test boostrapping error
        bootstrap_energy, standard_deviation = dmet.energy_error_bars(n_shots=50000, n_resamples=10, purify=False)
        rdm_measurements = copy(dmet.rdm_measurements)
        be_using_measurements, sd_using_measurements = dmet.energy_error_bars(n_shots=50000,
                                                                              n_resamples=10,
                                                                              purify=False,
                                                                              rdm_measurements=rdm_measurements)

        self.assertAlmostEqual(energy, -1.9916120594, delta=1e-3)
        # Should pass 99.993666% of the time with 4\sigma distance
        self.assertAlmostEqual(bootstrap_energy, -1.9916120594, delta=standard_deviation*4)
        self.assertAlmostEqual(be_using_measurements, -1.9916120594, delta=sd_using_measurements*4)

    def test_h4ring_vqe_resources(self):
        """Resources estimation on H4 ring."""

        opt_dmet = {"molecule": mol_H4_minao,
                    "fragment_atoms": [1, 1, 1, 1],
                    "fragment_solvers": ["vqe", "ccsd", "ccsd", "ccsd"],
                    "electron_localization": Localization.meta_lowdin,
                    "verbose": False
                    }

        # Building DMET fragments (with JW).
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        resources_jw = dmet.get_resources()

        # Building DMET fragments (with scBK).
        opt_dmet["solvers_options"] = {"qubit_mapping": "scbk", "initial_var_params": "ones", "up_then_down": True}
        dmet = DMETProblemDecomposition(opt_dmet)
        dmet.build()
        resources_bk = dmet.get_resources()

        # JW.
        self.assertEqual(resources_jw[0]["qubit_hamiltonian_terms"], 15)
        self.assertEqual(resources_jw[0]["circuit_width"], 4)
        # scBK.
        self.assertEqual(resources_bk[0]["qubit_hamiltonian_terms"], 5)
        self.assertEqual(resources_bk[0]["circuit_width"], 2)


if __name__ == "__main__":
    unittest.main()
