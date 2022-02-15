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

"""Test the functions in the main loop of DMET calculation."""

import unittest
import os
from pyscf import gto, scf
import numpy as np

from tangelo.problem_decomposition.dmet._helpers.dmet_orbitals import dmet_orbitals
from tangelo.problem_decomposition.dmet._helpers.dmet_onerdm import dmet_low_rdm, dmet_fragment_rdm
from tangelo.problem_decomposition.dmet._helpers.dmet_bath import dmet_fragment_bath
from tangelo.problem_decomposition.dmet._helpers.dmet_scf_guess import dmet_fragment_guess
from tangelo.problem_decomposition.dmet._helpers.dmet_scf import dmet_fragment_scf
from tangelo.problem_decomposition.electron_localization import iao_localization

path_file = os.path.dirname(os.path.abspath(__file__))


class TestDMETloop(unittest.TestCase):
    """Generate the localized orbitals employing IAOs."""

    def test_dmet_functions(self):

        # Initialize Molecule object with PySCF and input
        mol = gto.Mole()
        mol.atom = """
            C 0.94764 -0.02227  0.05901
            H 0.58322  0.35937 -0.89984
            H 0.54862  0.61702  0.85300
            H 0.54780 -1.03196  0.19694
            C 2.46782 -0.03097  0.07887
            H 2.83564  0.98716 -0.09384
            H 2.83464 -0.65291 -0.74596
            C 3.00694 -0.55965  1.40773
            H 2.63295 -1.57673  1.57731
            H 2.63329  0.06314  2.22967
            C 4.53625 -0.56666  1.42449
            H 4.91031  0.45032  1.25453
            H 4.90978 -1.19011  0.60302
            C 5.07544 -1.09527  2.75473
            H 4.70164 -2.11240  2.92450
            H 4.70170 -0.47206  3.57629
            C 6.60476 -1.10212  2.77147
            H 6.97868 -0.08532  2.60009
            H 6.97839 -1.72629  1.95057
            C 7.14410 -1.62861  4.10112
            H 6.77776 -2.64712  4.27473
            H 6.77598 -1.00636  4.92513
            C 8.66428 -1.63508  4.12154
            H 9.06449 -2.27473  3.32841
            H 9.02896 -2.01514  5.08095
            H 9.06273 -0.62500  3.98256
        """

        mol.basis = "3-21g"
        mol.charge = 0
        mol.spin = 0
        mol.build(verbose=0)

        # TODO: wrapper mean-field module
        mf = scf.RHF(mol)
        mf.scf()

        dmet_orbs = dmet_orbitals(mol, mf, range(mol.nao_nr()), iao_localization)

        # Test the construction of one particle RDM from low-level calculation
        onerdm_low = dmet_low_rdm(dmet_orbs.active_fock, dmet_orbs.number_active_electrons)
        onerdm_low_ref = np.loadtxt("{}/data/test_dmet_oneshot_loop_low_rdm.txt".format(path_file))
        for index, value_ref in np.ndenumerate(onerdm_low_ref):
            self.assertAlmostEqual(value_ref, onerdm_low[index], msg=f"Low-level RDM error at index {str(index)}",
                                   delta=1e-6)

        # Test the construction of bath orbitals
        t_list = [15]
        temp_list = [0, 15]
        chemical_potential = 0.0
        bath_orb, e_occupied = dmet_fragment_bath(dmet_orbs.mol_full, t_list, temp_list, onerdm_low)

        # Test the construction of one particle RDM for the fragment
        norb_high, nelec_high, onerdm_high = dmet_fragment_rdm(t_list, bath_orb, e_occupied, dmet_orbs.number_active_electrons)
        self.assertEqual(norb_high, 23, "The number of orbitals for a fragment does not agree")
        self.assertEqual(nelec_high, 16, "The number of electrons for a fragment does not agree")
        onerdm_high_ref = np.loadtxt("{}/data/test_dmet_oneshot_loop_core_rdm.txt".format(path_file))
        for index, value_ref in np.ndenumerate(onerdm_high_ref):
            self.assertAlmostEqual(value_ref, onerdm_high[index], msg="One RDM for fragment error at index " + str(index), delta=1e-6)

        # Test the construction of the Hamiltonian for the fragment
        one_ele, fock, two_ele = dmet_orbs.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

        # Test the construction of the guess orbitals for fragment SCF calculation
        guess_orbitals = dmet_fragment_guess(t_list, bath_orb, chemical_potential, norb_high, nelec_high, dmet_orbs.active_fock)

        # Test the fock matrix in the SCF calculation for a fragment
        mf_fragments, fock_frag_copy, mol = dmet_fragment_scf(t_list, two_ele, fock, nelec_high, norb_high, guess_orbitals, chemical_potential)

        # Test the energy calculation and construction of the one-particle RDM from the CC calculation for a fragment
        # fragment_energy, onerdm_frag, _, _ = dmet_fragment_cc_classical(mf_fragments, fock_frag_copy, t_list, one_ele, two_ele, fock)
        # self.assertAlmostEqual(fragment_energy, -82.70210049368914, msg="The DMET energy does no agree", delta=1e-6)


if __name__ == "__main__":
    unittest.main()
