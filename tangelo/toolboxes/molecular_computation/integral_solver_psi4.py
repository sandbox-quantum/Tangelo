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
        self.backend = psi4

    def set_physical_data(self, mol):
        self.backend.core.set_output_file('output.dat', False)
        if isinstance(mol.xyz, list):
            input_string = f"{mol.q} {mol.spin + 1} \n"
            for line in mol.xyz:
                input_string += f"{line[0]} {line[1][0]} {line[1][1]} {line[1][2]} \n"
            input_string += "symmetry c1"
            self.mol = self.backend.geometry(input_string)
        else:
            self.mol = self.backend.geometry(mol.xyz)
            mol.n_atoms = self.mol.natom()
            mol.xyz = list()
            for i in range(mol.n_atoms):
                mol.xyz += [(self.mol.symbol(i), tuple(self.mol.xyz(i)[p]*0.52917721067 for p in range(3)))]

        self.backend.set_options({'basis': "def2-msvp"})
        self.wfn = self.backend.core.Wavefunction.build(self.mol, self.backend.core.get_global_option('basis'))

        mol.n_electrons = self.wfn.nalpha() + self.wfn.nbeta()
        mol.n_atoms = self.mol.natom()

    def compute_mean_field(self, sqmol):
        # if sqmol.uhf:
        #    raise NotImplementedError(f"{self.__class__.__name__} does not currently support uhf")

        self.backend.set_options({'basis': sqmol.basis})
        if sqmol.uhf:
            self.backend.set_options({'reference': 'uhf', 'guess': 'gwh', 'guess_mix': True})
        elif sqmol.spin != 0:
            self.backend.set_options({'reference': 'rohf', 'guess': 'core'})
        else:
            self.backend.set_options({'reference': 'rhf'})

        sqmol.mf_energy, self.wfn = self.backend.energy('scf', molecule=self.mol, basis=sqmol.basis, return_wfn=True)
        self.mints = self.backend.core.MintsHelper(self.wfn.basisset())

        sqmol.mo_energies = np.asarray(self.wfn.epsilon_a())

        nbf = np.asarray(self.mints.ao_overlap()).shape[0]

        if sqmol.uhf:
            na = self.wfn.nalpha()
            nb = self.wfn.nbeta()
            sqmol.mo_occ = [[1]*na + (nbf-na)*[0]]+[[1]*nb + (nbf-nb)*[0]]
        else:
            docc = self.wfn.doccpi()[0]
            socc = self.wfn.soccpi()[0]
            sqmol.mo_occ = [2]*docc + [1]*socc + (nbf - docc - socc)*[0]
        sqmol.n_mos = nbf
        sqmol.n_sos = nbf*2
        self.mo_coeff = np.asarray(self.wfn.Ca()) if not sqmol.uhf else [np.asarray(self.wfn.Ca()), np.asarray(self.wfn.Cb())]
        self.ob = np.asarray(self.mints.ao_potential()) + np.asarray(self.mints.ao_kinetic())
        self.tb = np.asarray(self.mints.ao_eri())
        self.core_constant = self.mol.nuclear_repulsion_energy()

    def get_integrals(self, sqmol, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if sqmol.uhf:
            return self.compute_uhf_integrals(mo_coeff)

        ob = mo_coeff.T@self.ob@mo_coeff
        eed = self.tb.copy()
        eed = np.einsum("ij,jlmn -> ilmn", mo_coeff.T, eed)
        eed = np.einsum("kl,jlmn -> jkmn", mo_coeff.T, eed)
        eed = np.einsum("jlmn, mk -> jlkn", eed, mo_coeff)
        eed = np.einsum("jlmn, nk -> jlmk", eed, mo_coeff)
        tb = eed.transpose(0, 2, 3, 1)

        return self.core_constant, ob, tb

    def compute_uhf_integrals(self, mo_coeff):
        """Compute 1-electron and 2-electron integrals
        The return is formatted as
        [numpy.ndarray]*2 numpy array h_{pq} for alpha and beta blocks
        [numpy.ndarray]*3 numpy array storing h_{pqrs} for alpha-alpha, alpha-beta, beta-beta blocks

        Args:
            mo_coeff (List[array]): The molecular orbital coefficients for both spins [alpha, beta]

        Returns:
            List[array], List[array]: One and two body integrals
        """

        mo_a = self.backend.core.Matrix.from_array(mo_coeff[0])
        mo_b = self.backend.core.Matrix.from_array(mo_coeff[1])

        # calculate alpha and beta one-body integrals
        hpq = [mo_coeff[0].T.dot(self.ob).dot(mo_coeff[0]), mo_coeff[1].T.dot(self.ob).dot(mo_coeff[1])]

        # mo transform the two-electron integrals
        eri_a = np.asarray(self.mints.mo_eri(mo_a, mo_a, mo_a, mo_a))
        eri_b = np.asarray(self.mints.mo_eri(mo_b, mo_b, mo_b, mo_b))
        eri_ba = np.asarray(self.mints.mo_eri(mo_a, mo_a, mo_b, mo_b))

        # # convert this into the order OpenFemion like to receive
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Gpqrs has alpha, alphaBeta, Beta blocks
        Gpqrs = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        return self.core_constant, hpq, Gpqrs
