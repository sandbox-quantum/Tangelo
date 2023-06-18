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

from copy import copy

import numpy as np

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


class IntegralSolverPsi4(IntegralSolver):
    """psi4 IntegrationSolver class"""
    def __init__(self):
        import psi4
        self.backend = psi4
        self.backend.core.clean()
        self.backend.core.clean_options()
        self.backend.core.clean_variables()

    def set_physical_data(self, mol):
        """Set molecular data that is independant of basis set in mol

        Modify mol variable:
            mol.xyz to (list): Nested array-like structure with elements and coordinates
                                            (ex:[ ["H", (0., 0., 0.)], ...]) in angstrom
        Add to mol:
            mol.n_electrons (int): Self-explanatory.
            mol.n_atoms (int): Self-explanatory.

        Args:
            mol (Molecule or SecondQuantizedMolecule): Class to add the other variables given populated.
                mol.xyz (in appropriate format for solver): Definition of molecular geometry.
                mol.q (float): Total charge.
                mol.spin (int): Absolute difference between alpha and beta electron number.
        """
        self.backend.core.set_output_file('output.dat', False)
        if isinstance(mol.xyz, list):
            input_string = f"{mol.q} {mol.spin + 1} \n"
            for line in mol.xyz:
                input_string += f"{line[0]} {line[1][0]} {line[1][1]} {line[1][2]} \n"
            input_string += "symmetry c1"
            self.mol = self.backend.geometry(input_string)
            self.mol_nosym = self.backend.geometry(input_string)
        else:
            self.mol = self.backend.geometry(mol.xyz)
            mol.n_atoms = self.mol.natom()
            mol.xyz = list()
            for i in range(mol.n_atoms):
                mol.xyz += [(self.mol.symbol(i), tuple(self.mol.xyz(i)[p]*0.52917721067 for p in range(3)))]

        self.backend.set_options({'basis': "def2-msvp"})
        wfn = self.backend.core.Wavefunction.build(self.mol, self.backend.core.get_global_option('basis'))

        mol.n_electrons = wfn.nalpha() + wfn.nbeta()
        mol.n_atoms = self.mol.natom()

    def compute_mean_field(self, sqmol):
        """Run a unrestricted/restricted (openshell-)Hartree-Fock calculation and modify/add the following
        variables to sqmol

        Modify sqmol variables.
            sqmol.mf_energy (float): Mean-field energy (RHF or ROHF energy depending on the spin).
            sqmol.mo_energies (list of float): Molecular orbital energies.
            sqmol.mo_occ (list of float): Molecular orbital occupancies (between 0. and 2.).
            sqmol.n_mos (int): Number of molecular orbitals with a given basis set.
            sqmol.n_sos (int): Number of spin-orbitals with a given basis set.

        Add to sqmol:
            self.mo_coeff (ndarray or List[ndarray]): array of molecular orbital coefficients (MO coeffs) if RHF ROHF
                                                        list of arrays [alpha MO coeffs, beta MO coeffs] if UHF

        Args:
            sqmol (SecondQuantizedMolecule): Populated variables of Molecule plus
                sqmol.basis (string): Basis set.
                sqmol.ecp (dict): The effective core potential (ecp) for any atoms in the molecule.
                    e.g. {"C": "crenbl"} use CRENBL ecp for Carbon atoms.
                sqmol.symmetry (bool or str): Whether to use symmetry in RHF or ROHF calculation.
                    Can also specify point group using string. e.g. "Dooh", "D2h", "C2v", ...
                sqmol.uhf (bool): If True, Use UHF instead of RHF or ROHF reference. Default False


        """
        if sqmol.symmetry:
            input_string = f"{sqmol.q} {sqmol.spin + 1} \n"
            for line in sqmol.xyz:
                input_string += f"{line[0]} {line[1][0]} {line[1][1]} {line[1][2]} \n"
            if isinstance(sqmol.symmetry, str):
                input_string += "symmetry" + sqmol.symmetry
            self.mol = self.backend.geometry(input_string)

        self.backend.set_options({'basis': sqmol.basis})
        if sqmol.uhf:
            self.backend.set_options({'reference': 'uhf', 'guess': 'gwh', 'guess_mix': True})
        elif sqmol.spin != 0:
            self.backend.set_options({'reference': 'rohf', 'guess': 'core'})
        else:
            self.backend.set_options({'reference': 'rhf'})

        sqmol.mf_energy, self.sym_wfn = self.backend.energy('scf', molecule=self.mol, basis=self.backend.core.get_global_option('basis'), return_wfn=True)
        self.wfn = self.sym_wfn.c1_deep_copy(self.sym_wfn.basisset())
        self.backend.core.clean_options()

        sqmol.mo_energies = np.asarray(self.wfn.epsilon_a())
        if sqmol.symmetry:
            self.irreps = [self.mol.point_group().char_table().gamma(i).symbol().upper() for i in range(self.sym_wfn.nirrep())]
            sym_mo_energies = []
            tmp = self.backend.driver.p4util.numpy_helper._to_array(self.sym_wfn.epsilon_a(), dense=False)
            for i in self.irreps:
                sym_mo_energies += [(i, j, x) for j, x in enumerate(tmp[self.irreps.index(i)])]
            ordered_energies = sorted(sym_mo_energies, key=lambda x: x[1])
            sqmol.mo_symm_labels = [o[0] for o in ordered_energies]
            sqmol.mo_symm_ids = [o[1] for o in ordered_energies]
        else:
            sqmol.mo_symm_labels = None
            sqmol.mo_symm_ids = None

        self.mints = self.backend.core.MintsHelper(self.wfn.basisset())
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
        r"""Computes core constant, one_body, and two-body integrals for all orbitals

        one-body integrals should be in the form
        h[p,q]= \int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx

        two-body integrals should be in the form
        h[p,q,r,s] = \int \phi_p(x) * \phi_q(y) * V_{elec-elec} \phi_r(y) \phi_s(x) dxdy

        Using molecular orbitals \phi_j(x) = \sum_{ij} A_i(x) mo_coeff_{i,j} where A_i(x) are the atomic orbitals.

        For UHF (if sqmol.uhf is True)
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        where one_body and two_body are appropriately sized arrays for each spin sector.

        Args:
            sqmol (SecondQuantizedMolecule) : SecondQuantizedMolecule populated with all variables defined above
            mo_coeff : Molecular orbital coefficients to use for calculating the integrals, instead of self.mo_coeff

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """
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

        # # convert this into physicist ordering for OpenFermion
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Gpqrs has alpha, alphaBeta, Beta blocks
        Gpqrs = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        return self.core_constant, hpq, Gpqrs
