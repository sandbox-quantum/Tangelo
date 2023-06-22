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

"""Construct localized orbitals for DMET calculation

The construction of the localized orbitals from the orbitals obtained from the
full system mean-field calculation is done here.

Localization schemes based on Intrisic atomic orbitals (IAO) (dmet_iao.py),
Lowdin, meta-Lowdin, and Boys can be selected. The latter three rely on external
PySCF program. However, using Boys localization is not recommended.
"""

import numpy as np


class dmet_orbitals:
    """ Localize the SCF orbitals and calculate the integrals

    This class handles the information from the calculation of the entire
    molecular system

    Attributes:
        mol_full (pyscf.gto.Mole): The molecule to simulate (The full molecule).
        mf_full (pyscf.scf.RHF): The mean-field of the molecule (The full
            molecule).
        low_scf_energy (float64): The mean-field total energy of the full
            molecule.
        low_scf_fock (numpy.array): The Fock matrix for the full molecule
            (float64).
        dmet_active_orbitals (list): Label for active orbitals. "1" is active,
            and "0" is inactive (int).
        number_active_orbitals (int): Number of active orbitals.
        number_active_electrons (int): Number of active electrons.
        localized_mo (numpy.array): The localized molecular orbitals (float64).
        core_constant_energy (float64): Constant core energy (Constant value
            which does not change in DMET).
        active_oneint (numpy.array): One-electron integral in localized MO basis
            (float64).
        active_fock (numpy.array): Fock matrix in localized MO basis (float64).
        uhf (bool): Flag for an unrestricted mean-field.
    """

    def __init__(self, mol, mf, active_space, localization_function, uhf):
        """Initialize the class.

        Localize the orbitals, get energies and integrals for the entire system
        in localized MO basis.

        Args:
            mol (pyscf.gto.Mole): The molecule to simulate (The full molecule).
            mf (pyscf.scf.RHF): The mean field of the molecule (The full
                molecule).
            active_space (list): The active space in DMET calculation. All
                orbitals in the initial SCF calculation (int).
            localization_function (string): Localization scheme.
            uhf (bool): Flag for an unrestricted mean-field.
        """
        from pyscf import scf, ao2mo
        self.pyscfscf = scf
        self.pyscfao2mo = ao2mo

        # General quantities.
        self.mol_full = mol
        self.mf_full = mf
        self.low_scf_energy = mf.e_tot

        # Define the active space if possible.
        self.dmet_active_orbitals = np.zeros([mf.mol.nao_nr()], dtype=int)
        self.dmet_active_orbitals[active_space] = 1
        self.number_active_orbitals = np.sum(self.dmet_active_orbitals)

        # Localize the orbitals.
        self.localized_mo = localization_function(mol, mf)

        if uhf:
            self._unrestricted_init()
        else:
            self._restricted_init()

    def _restricted_init(self):
        """Initialize the attributes for a restricted mean-field."""

        self.number_active_electrons = int(np.rint(self.mf_full.mol.nelectron - np.sum(self.mf_full.mo_occ[self.dmet_active_orbitals == 0])))

        # RHF
        if self.mol_full.spin == 0:
            # Obtain the elements from the low-level SCF calculations.
            low_scf_dm = self.mf_full.mo_coeff @ np.diag(self.mf_full.mo_occ) @ self.mf_full.mo_coeff.T
            low_scf_twoint = self.pyscfscf.hf.get_veff(self.mf_full.mol, low_scf_dm, 0, 0, 1)
            self.low_scf_fock = self.mf_full.mol.intor("cint1e_kin_sph") + self.mf_full.mol.intor("cint1e_nuc_sph") + low_scf_twoint
            # Add effective core potential to Fock matrix if applicable.
            if len(self.mol_full._ecpbas) > 0:
                self.low_scf_fock += self.mf_full.mol.intor_symmetric('ECPscalar')

            # Define the core space if possible (Initial calculations treat the entire molecule ...).
            core_mo_dm = np.array(self.mf_full.mo_occ, copy=True)
            core_mo_dm[self.dmet_active_orbitals == 1] = 0
            core_ao_dm = self.mf_full.mo_coeff @ np.diag(core_mo_dm) @ self.mf_full.mo_coeff.T
            core_twoint = self.pyscfscf.hf.get_veff(self.mf_full.mol, core_ao_dm, 0, 0, 1)
            core_oneint = self.low_scf_fock - low_scf_twoint + core_twoint

            # Define the energies and matrix elements based on the localized orbitals.
            self.core_constant_energy = self.mf_full.mol.energy_nuc() + np.einsum("ij,ij->", core_oneint - 0.5*core_twoint, core_ao_dm)
            self.active_oneint = self.localized_mo.T @ core_oneint @ self.localized_mo
            self.active_fock = self.localized_mo.T @ self.low_scf_fock @ self.localized_mo
        # ROHF
        else:
            # Obtain the elements from the low-level SCF calculations.
            low_scf_rdm = self.mf_full.make_rdm1()
            low_scf_twoint = self.mf_full.get_veff(self.mol_full, low_scf_rdm, 0, 0, 1)

            core_oneint = self.mf_full.get_hcore()
            low_scf_fock_alpha = core_oneint + low_scf_twoint[0]
            low_scf_fock_beta = core_oneint + low_scf_twoint[1]

            elec_paired = self.number_active_electrons - self.mol_full.spin
            orbital_paired = elec_paired // 2
            self.number_active_electrons_alpha = orbital_paired + self.mol_full.spin
            self.number_active_electrons_beta = orbital_paired

            # Define the energies and matrix elements based on the localized orbitals.
            self.core_constant_energy = self.mf_full.mol.energy_nuc()
            self.active_oneint = self.localized_mo.T @ core_oneint @ self.localized_mo

            self.active_fock_alpha = self.localized_mo.T @ low_scf_fock_alpha @ self.localized_mo
            self.active_fock_beta = self.localized_mo.T @ low_scf_fock_beta @ self.localized_mo

            rdm_a = self.localized_mo.T @ low_scf_rdm[0] @ self.localized_mo
            rdm_b = self.localized_mo.T @ low_scf_rdm[1] @ self.localized_mo
            rdm_total = np.array((rdm_a, rdm_b))

            overlap = np.eye(self.number_active_orbitals)
            two_int = self.pyscfscf.hf.get_veff(self.mol_full, rdm_total, 0, 0, 1)
            new_fock_alpha = self.active_oneint + (self.localized_mo.T @ two_int[0] @ self.localized_mo)
            new_fock_beta = self.active_oneint + (self.localized_mo.T @ two_int[1] @ self.localized_mo)
            fock_total = np.array((new_fock_alpha, new_fock_beta))
            self.active_fock = self.pyscfscf.rohf.get_roothaan_fock(fock_total, rdm_total, overlap)

    def _unrestricted_init(self):
        """Initialize the attributes for an unrestricted mean-field."""

        low_scf_fock_alpha, low_scf_fock_beta = self.mf_full.get_fock()
        core_oneint = self.mf_full.get_hcore()

        self.number_active_electrons = self.mf_full.mol.nelectron

        elec_diff = self.mol_full.spin
        elec_paired = self.number_active_electrons-elec_diff
        orbital_paired = elec_paired // 2
        self.number_active_electrons_alpha = orbital_paired + elec_diff
        self.number_active_electrons_beta = orbital_paired

        self.core_constant_energy = self.mf_full.mol.energy_nuc()
        self.active_oneint = self.localized_mo.T @ core_oneint @ self.localized_mo

        self.active_fock_alpha = self.localized_mo.T @ low_scf_fock_alpha @ self.localized_mo
        self.active_fock_beta = self.localized_mo.T @ low_scf_fock_beta @ self.localized_mo

    def dmet_fragment_hamiltonian(self, bath_orb, norb_high, onerdm_core):
        """Construct the Hamiltonian for a DMET fragment.

        Args:
            bath_orb (numpy.array): The bath orbitals (float64).
            norb_high (int): Number of orbitals in the fragment calculation.
            onerdm_core (numpy.array): The core part of the one-particle RDM
                (float64).

        Returns:
            numpy.array: One-electron integrals for fragment calculation
                (float64).
            numpy.array: The fock matrix for fragment calculation (float64).
            numpy.array: Two-electron integrals for fragment calculation
                (float64).
        """

        # Calculate one-electron integrals.
        frag_oneint = bath_orb[:, : norb_high].T @ self.active_oneint @ bath_orb[:, : norb_high]

        # Calculate the fock matrix.
        density_matrix = self.localized_mo @ onerdm_core @ self.localized_mo.T
        two_int = self.pyscfscf.hf.get_veff(self.mol_full, density_matrix, 0, 0, 1)
        new_fock = self.active_oneint + (self.localized_mo.T @ two_int @ self.localized_mo)
        frag_fock = bath_orb[:, : norb_high].T @ new_fock @ bath_orb[:, : norb_high]

        # Calculate the two-electron integrals.
        coefficients = np.dot(self.localized_mo, bath_orb[:, : norb_high])
        frag_twoint = self.pyscfao2mo.outcore.full_iofree(self.mol_full, coefficients, compact=False).reshape(
                                                norb_high,  norb_high,  norb_high,  norb_high)

        return frag_oneint, frag_fock, frag_twoint
