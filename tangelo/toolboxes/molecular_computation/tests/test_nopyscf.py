# Copyright 2023 Good Chemistry Company.
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

from tangelo import SecondQuantizedMolecule
from tangelo.helpers.utils import installed_chem_backends
from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver
from tangelo.algorithms.variational import SA_OO_Solver, BuiltInAnsatze, ADAPTSolver

h2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]


class IntegralSolverDummy(IntegralSolver):
    def set_physical_data(self, mol):
        mol.n_electrons = 2
        mol.n_atoms = 2

    def compute_mean_field(self, sqmol):
        npzfile = np.load(os.path.dirname(os.path.abspath(__file__))+'/data/h2_631g.npz')
        sqmol.mf_energy = npzfile['mf_energy']
        sqmol.mo_energies = npzfile['mo_energies']
        sqmol.mo_occ = npzfile['mo_occ']
        sqmol.n_mos = npzfile['n_mos']
        sqmol.n_sos = npzfile['n_sos']
        self.mo_coeff = npzfile['mo_coeff']
        self.inv_mo = np.linalg.inv(self.mo_coeff)
        self.ob = npzfile['one_body']
        self.tb = npzfile['two_body']
        self.core_constant = float(npzfile['core_constant'])

    def get_integrals(self, sqmol=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        # Any proper mo_coeff should be a unitary transformation of the original mo_coeff
        # apply inverse of original mo_coeff to obtain that unitary to apply.
        isq = self.inv_mo@mo_coeff

        # Calculate new integrals
        if np.allclose(isq, np.eye(isq.shape[0])):
            ob = self.ob
            tb = self.tb
        else:
            ob = isq.T@self.ob@isq
            eed1 = self.tb.transpose(0, 3, 1, 2)
            eed1 = np.einsum("ij,jlmn -> ilmn", isq.T, eed1)
            eed1 = np.einsum("kl,jlmn -> jkmn", isq.T, eed1)
            eed1 = np.einsum("jlmn, mk -> jlkn", eed1, isq)
            eed = np.einsum("jlmn, nk -> jlmk", eed1, isq)
            tb = eed.transpose(0, 2, 3, 1)

        return self.core_constant, ob, tb


class TestNopyscf(unittest.TestCase):
    @unittest.skipIf(len(installed_chem_backends) > 0, "Test Skipped: A chem backend is available \n")
    def test_no_solver(self):
        "Test that a ValueError is raised when SecondQuantizedMolecule is called without PySCF, Psi4 or custom solver"
        with self.assertRaises(ValueError):
            SecondQuantizedMolecule(h2, 0, 0)

    def test_sa_oo_vqe(self):
        "Test that sa_oo_vqe works properly when using a user defined IntegralSolver that only reads in integrals"
        molecule_dummy = SecondQuantizedMolecule(h2, 0, 0, IntegralSolverDummy(), basis="6-31g", frozen_orbitals=[3])
        sa_oo_vqe = SA_OO_Solver({"molecule": molecule_dummy, "ref_states": [[1, 1, 0, 0, 0, 0]],
                                  "tol": 1.e-5, "ansatz": BuiltInAnsatze.UCCSD, "n_oo_per_iter": 25,
                                  "initial_var_params": [1.e-5]*5})
        sa_oo_vqe.build()
        sa_oo_vqe.iterate()

        self.assertAlmostEqual(sa_oo_vqe.state_energies[0], -1.15137, places=4)

    def test_adapt_vqe_solver(self):
        "Test that ADAPTVQE works with a user defined IntegralSolver."
        molecule_dummy = SecondQuantizedMolecule(h2, 0, 0, IntegralSolverDummy(), basis="6-31g", frozen_orbitals=[])

        adapt_vqe = ADAPTSolver({"molecule": molecule_dummy})
        adapt_vqe.build()
        energy = adapt_vqe.simulate()
        self.assertAlmostEqual(energy, -1.15168, places=4)


if __name__ == "__main__":
    unittest.main()
