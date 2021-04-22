#   Copyright 2019 1QBit
#   
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Construct localized orbitals for DMET calculation

The construction of the localized orbitals from the orbitals 
obtained from the full system mean-field calculation is done here.

Localization schemes based on Intrisic atomic orbitals (IAO)
(dmet_iao.py), Lowdin, meta-Lowdin, and Boys can be selected.
The latter three rely on external PySCF program.
However, using Boys localization is not recommended. 

"""

from pyscf import scf, ao2mo
import numpy as np
from functools import reduce

class dmet_orbitals:
    """ Localize the SCF orbitals and calculate the integrals

    This class handles the information from the calculation 
    of the entire molecular system

    Attributes:
        mol_full (pyscf.gto.Mole): The molecule to simulate (The full molecule).
        mf_full (pyscf.scf.RHF): The mean-field of the molecule (The full molecule).
        low_scf_energy (float64): The mean-field total energy of the full molecule.
        low_scf_fock (numpy.array): The Fock matrix for the full molecule (float64).
        dmet_active_orbitals (list): Label for active orbitals. "1" is active, and "0" is inactive (int).
        number_active_orbitals (int): Number of active orbitals.
        number_active_electrons (int): Number of active electrons.
        localized_mo (numpy.array): The localized molecular orbitals (float64).
        core_constant_energy (float64): Constant core energy (Constant value which does not change in DMET).
        active_oneint (numpy.array): One-electron integral in localized MO basis (float64).
        active_fock (numpy.array): Fock matrix in localized MO basis (float64).
    """

    def __init__(self, mol, mf, active_space, localization_function):
        """Initialize the class.

        Localize the orbitals, get energies and integrals for the entire system
        in localized MO basis.

        Args: 
            mol (pyscf.gto.Mole): The molecule to simulate (The full molecule).
            mf (pyscf.scf.RHF): The mean field of the molecule (The full molecule).
            active_space (list): The active space in DMET calculation. All orbitals in the initial SCF calculation (int).
            localization_function (string): Localization scheme.
        """

        # TODO: Is active space always claculated from the molecule?

        # Obtain the elements from the low-level SCF calculations
        self.mol_full = mol
        self.mf_full = mf
        self.low_scf_energy = mf.e_tot
        low_scf_dm = reduce(np.dot, (mf.mo_coeff, np.diag(mf.mo_occ), mf.mo_coeff.T))
        low_scf_twoint = scf.hf.get_veff(mf.mol, low_scf_dm, 0, 0, 1)
        self.low_scf_fock = mf.mol.intor('cint1e_kin_sph') + mf.mol.intor('cint1e_nuc_sph') + low_scf_twoint

        # Define the active space if possible
        self.dmet_active_orbitals = np.zeros([mf.mol.nao_nr()], dtype=int)
        self.dmet_active_orbitals[active_space] = 1
        self.number_active_orbitals = np.sum(self.dmet_active_orbitals)
        self.number_active_electrons = int(np.rint(mf.mol.nelectron - np.sum(mf.mo_occ[self.dmet_active_orbitals==0])))

        # Localize the orbitals (IAO)
        self.localized_mo = localization_function(mol, mf)

        # Define the core space if possible (Initial calculations treat the entire molecule ...)
        core_mo_dm = np.array(mf.mo_occ, copy=True)
        core_mo_dm[self.dmet_active_orbitals == 1] = 0
        core_ao_dm = reduce(np.dot, (mf.mo_coeff, np.diag(core_mo_dm), mf.mo_coeff.T))
        core_twoint = scf.hf.get_veff(mf.mol, core_ao_dm, 0, 0, 1)
        core_oneint = self.low_scf_fock - low_scf_twoint + core_twoint

        # Define the energies and matrix elements based on the localized orbitals
        self.core_constant_energy = mf.mol.energy_nuc() + np.einsum('ij,ij->', core_oneint - 0.5*core_twoint, core_ao_dm)
        self.active_oneint = reduce(np.dot, (self.localized_mo.T, core_oneint, self.localized_mo))
        self.active_fock = reduce(np.dot, (self.localized_mo.T, self.low_scf_fock, self.localized_mo))

    def dmet_fragment_hamiltonian(self, bath_orb, norb_high, onerdm_core):
        """Construct the Hamiltonian for a DMET fragment.

        Args:
            bath_orb (numpy.array): The bath orbitals (float64).
            norb_high (int): Number of orbitals in the fragment calculation.
            onerdm_core (numpy.array): The core part of the one-particle RDM (float64).

        Returns:
            frag_oneint (numpy.array): One-electron integrals for fragment calculation (float64).
            frag_fock (numpy.array): The fock matrix for fragment calculation (float64).
            frag_twoint (numpy.array): Two-electron integrals for fragment calculation (float64).
        """

        # Calculate one-electron integrals
        frag_oneint = reduce(np.dot, (bath_orb[ : , : norb_high].T, self.active_oneint, bath_orb[ : , : norb_high]))

        # Calculate the fock matrix
        density_matrix = reduce(np.dot, (self.localized_mo, onerdm_core, self.localized_mo.T))
        two_int = scf.hf.get_veff(self.mol_full, density_matrix, 0, 0, 1)
        new_fock = self.active_oneint + reduce(np.dot, ((self.localized_mo.T, two_int, self.localized_mo)))
        frag_fock = reduce(np.dot, (bath_orb[ : , : norb_high ].T, new_fock, bath_orb[ : , : norb_high]))

        # Calculate the two-electron integrals
        coefficients = np.dot(self.localized_mo, bath_orb[ : , : norb_high])
        frag_twoint = ao2mo.outcore.full_iofree(self.mol_full, coefficients, compact=False).reshape( \
                                                norb_high,  norb_high,  norb_high,  norb_high)

        return frag_oneint, frag_fock, frag_twoint

