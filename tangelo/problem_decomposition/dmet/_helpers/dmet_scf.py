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

"""Perform fragment SCF calculation.

The fragment SCF calculation for DMET calculation is done here.
"""

import numpy as np


def dmet_fragment_scf_rhf(t_list, two_ele, fock, n_electrons, n_orbitals, guess_orbitals, chemical_potential):
    """Perform SCF calculation.

    Args:
        t_list (list): Number of [0] fragment & [1] bath orbitals (int).
        two_ele (numpy.array): Two-electron integrals for fragment calculation
            (float64).
        fock (numpy.array): The fock matrix for fragment calculation (float64).
        n_electrons (int): Number of electrons for fragment calculation.
        n_orbitals (int): Number of orbitals for fragment calculation.
        guess_orbitals (numpy.array): Guess orbitals for SCF calculation
            (float64).
        chemical_potential (float64): The chemical potential.

    Returns:
        pyscf.scf.RHF: The mean field of the molecule (Fragment calculation).
        numpy.array: The fock matrix with chemical potential subtracted
            (float64).
        pyscf.gto.Mole: The molecule to simulate (Fragment calculation).
    """
    from pyscf import gto, scf, ao2mo
    # Deep copy the fock matrix
    fock_frag_copy = fock.copy()

    # Subtract the chemical potential to make the number of electrons consistent
    for orb in range(t_list[0]):
        fock_frag_copy[orb, orb] -= chemical_potential

    # Determine the molecular space (set molecule object of pyscf)
    mol_frag = gto.Mole()
    mol_frag.build(verbose=0)
    mol_frag.atom.append(("C", (0, 0, 0)))
    mol_frag.nelectron = n_electrons
    mol_frag.incore_anyway = True

    # Perform SCF calculation (set mean field object of pyscf)
    mf_frag = scf.RHF(mol_frag)
    mf_frag.get_hcore = lambda *args: fock_frag_copy
    mf_frag.get_ovlp = lambda *args: np.eye(n_orbitals)
    mf_frag._eri = ao2mo.restore(8, two_ele, n_orbitals)
    mf_frag.scf(guess_orbitals)

    # Use newton-raphson algorithm if the above SCF calculation is not converged
    if (mf_frag.converged is False):
        mf_frag = scf.RHF(mol_frag).newton()

    return mf_frag, fock_frag_copy, mol_frag


def dmet_fragment_scf_rohf_uhf(nele_ab, two_ele, fock, n_electrons, n_orbitals, guess_orbitals, chemical_potential, uhf):
    """Perform SCF calculation.

    Args:
        nele_ab (list): List of the alpha and beta electron number (int).
        two_ele (numpy.array): Two-electron integrals for fragment calculation
            (float64).
        fock (numpy.array): The fock matrix for fragment calculation (float64).
        n_electrons (int): Number of electrons for fragment calculation.
        n_orbitals (int): Number of orbitals for fragment calculation.
        guess_orbitals (numpy.array): Guess orbitals for SCF calculation
            (float64).
        chemical_potential (float64): The chemical potential.
        uhf (bool): Flag for UHF mean-field. If not, a ROHF mean-field is
            considered.

    Returns:
        pyscf.scf.ROHF or pyscf.scf.UHF: The mean field of the molecule
            (Fragment calculation).
        numpy.array: The fock matrix with chemical potential subtracted
            (float64).
        pyscf.gto.Mole: The molecule to simulate (Fragment calculation).
    """
    from pyscf import gto, scf, ao2mo

    # Deep copy the fock matrix
    fock_frag_copy = fock.copy()

    for orb in range(nele_ab[0]):
        fock_frag_copy[orb, orb] -= chemical_potential

    # Determine the molecular space (set molecule object of pyscf)
    mol_frag = gto.Mole()
    mol_frag.build(verbose=0)
    mol_frag.atom.append(("C", (0, 0, 0)))
    mol_frag.nelectron = n_electrons
    mol_frag.incore_anyway = True
    mol_frag.spin = nele_ab[0] - nele_ab[1]

    # Perform SCF calculation (set mean field object of pyscf)
    mf_frag = scf.UHF(mol_frag) if uhf else scf.ROHF(mol_frag)
    mf_frag.get_hcore = lambda *args: fock_frag_copy
    mf_frag.get_ovlp = lambda *args: np.eye(n_orbitals)
    mf_frag._eri = ao2mo.restore(8, two_ele, n_orbitals)
    mf_frag.scf(guess_orbitals)

    orb_temp = mf_frag.mo_coeff
    occ_temp = mf_frag.mo_occ

    # Use Newton-Raphson algorithm if the above SCF calculation is not converged
    if not mf_frag.converged:
        mf_frag = scf.newton(mf_frag)
        _ = mf_frag.kernel(orb_temp, occ_temp)

    return mf_frag, fock_frag_copy, mol_frag
