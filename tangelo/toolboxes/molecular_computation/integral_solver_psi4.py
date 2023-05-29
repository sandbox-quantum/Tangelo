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

import numpy as np

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


class IntegralSolverPsi4(IntegralSolver):
    """psi4 IntegrationSolver class"""
    def __init__(self):
        import psi4
        self.psi4 = psi4

    def set_physical_data(self, mol):
        self.psi4.core.set_output_file('output.dat', False)
        if isinstance(mol.xyz, list):
            input_string = f"{mol.q} {mol.spin + 1} \n"
            for line in mol.xyz:
                input_string += f"{line[0]} {line[1][0]} {line[1][1]} {line[1][2]} \n"
            input_string += "symmetry c1"
            self.mol = self.psi4.geometry(input_string)
        else:
            self.mol = self.psi4.geometry(mol.xyz)
            mol.n_atoms = self.mol.natom()
            mol.xyz = list()
            for i in range(mol.n_atoms):
                mol.xyz += [(self.mol.symbol(i), tuple(self.mol.xyz(i)[p]*0.52917721067 for p in range(3)))]

        self.psi4.set_options({'basis': "def2-msvp"})
        self.wfn = self.psi4.core.Wavefunction.build(self.mol, self.psi4.core.get_global_option('basis'))

        mol.n_electrons = self.wfn.nalpha() + self.wfn.nbeta()
        mol.n_atoms = self.mol.natom()

    def compute_mean_field(self, sqmol):
        if sqmol.uhf:
            raise NotImplementedError(f"{self.__class__.__name__} does not currently support uhf")

        self.psi4.set_options({'basis': sqmol.basis})
        if sqmol.spin != 0:
            self.psi4.set_options({'reference': 'rohf'})

        sqmol.mf_energy, self.wfn = self.psi4.energy('scf', molecule=self.mol, basis=sqmol.basis, return_wfn=True)
        mints = self.psi4.core.MintsHelper(self.wfn.basisset())

        sqmol.mo_energies = np.asarray(self.wfn.epsilon_a())

        nbf = np.asarray(mints.ao_overlap()).shape[0]
        docc = self.wfn.doccpi()[0]
        socc = self.wfn.soccpi()[0]
        sqmol.mo_occ = [2]*docc + [1]*socc + (nbf - docc - socc)*[0]
        sqmol.n_mos = nbf
        sqmol.n_sos = nbf*2
        self.mo_coeff = np.asarray(self.wfn.Ca())
        self.ob = np.asarray(mints.ao_potential()) + np.asarray(mints.ao_kinetic())
        self.tb = np.asarray(mints.ao_eri())
        self.core_constant = self.mol.nuclear_repulsion_energy()

    def get_integrals(self, sqmol, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        ob = mo_coeff.T@self.ob@mo_coeff
        eed = self.tb.copy()
        eed = np.einsum("ij,jlmn -> ilmn", mo_coeff.T, eed)
        eed = np.einsum("kl,jlmn -> jkmn", mo_coeff.T, eed)
        eed = np.einsum("jlmn, mk -> jlkn", eed, mo_coeff)
        eed = np.einsum("jlmn, nk -> jlmk", eed, mo_coeff)
        tb = eed.transpose(0, 2, 3, 1)

        return self.core_constant, ob, tb
