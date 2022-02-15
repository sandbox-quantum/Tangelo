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

"""Perform IAO localization.

The orbital localization of the canonical orbitals using Intrinsic Atomic
Orbitals (IAO) localization is done here. `pyscf.lo` is used.

Note that minimal basis cannot be used for IAO because the idea of IAO is to map
on minao minimal basis set.

For details, refer to:
    - G. Knizia, JCTC 9, 4834-4843 (2013).
"""

from pyscf import gto
from pyscf.lo import iao
from functools import reduce
from pyscf.lo import orth
import numpy as np
import scipy


def iao_localization(mol, mf):
    """Localize the orbitals using IAO localization.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        mf (pyscf.scf.RHF): The mean field of the molecule.

    Returns:
        numpy.array: The localized orbitals (float64).
    """

    if mol.basis in {"minao", "sto-3g", "sto-6g"}:
        raise RuntimeError("Using IAO localization with minimal basis is not supported.")

    #   Construct IAO from occupied orbitals
    iao1 = _iao_occupied_orbitals(mol, mf)

    #   Construct IAO from complementary space
    iao2 = _iao_complementary_orbitals(mol, iao1)

    #   Gather two and assign the IAOs to atoms, rearrange them
    iao_lo = _iao_atoms(mol, iao1, iao2)

    return iao_lo


def _iao_occupied_orbitals(mol, mf):
    """Get the IAOs for occupied space.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        mf (pyscf.scf.RHF): The mean field of the molecule.

    Returns:
        numpy.array: The localized orbitals for the occupied space (float64).
    """

    #   Get MO coefficient of occupied MOs
    occupied_orbitals = mf.mo_coeff[:, mf.mo_occ > 0.5]

    #   Get mol data in minao basis
    min_mol = iao.reference_mol(mol)

    #   Calculate the overlaps for total basis
    s1 = mol.intor_symmetric("int1e_ovlp")

    #   ... for minao basis
    s2 = min_mol.intor_symmetric("int1e_ovlp")

    #   ... between the two basis (and transpose)
    s12 = gto.mole.intor_cross("int1e_ovlp", mol, min_mol)
    s21 = s12.T

    #   Calculate P_12 = S_1^-1 * S_12 using Cholesky decomposition
    s1_sqrt = scipy.linalg.cho_factor(s1)
    s2_sqrt = scipy.linalg.cho_factor(s2)
    p12 = scipy.linalg.cho_solve(s1_sqrt, s12)

    #   C~ = second_half ( S_1^-1 * S_12 * first_half ( S_2^-1 * S_21 * C ) )
    c_tilde = scipy.linalg.cho_solve(s2_sqrt, np.dot(s21, occupied_orbitals))
    c_tilde = scipy.linalg.cho_solve(s1_sqrt, np.dot(s12, c_tilde))
    c_tilde = np.dot(c_tilde, orth.lowdin(reduce(np.dot, (c_tilde.T, s1, c_tilde))))

    #   Obtain C * C^T * S1 and C~ * C~^T * S1
    ccs1 = reduce(np.dot, (occupied_orbitals, occupied_orbitals.conj().T, s1))
    ctcts1 = reduce(np.dot, (c_tilde, c_tilde.conj().T, s1))

    #   Calculate A = ccs1 * ctcts1 * p12 + ( 1 - ccs1 ) * ( 1 - ctcts1 ) * p12
    iao_active = (p12 + reduce(np.dot, (ccs1, ctcts1, p12)) * 2 - np.dot(ccs1, p12) - np.dot(ctcts1, p12))

    #   Orthogonalize A
    iao_active = np.dot(iao_active, orth.lowdin(reduce(np.dot, (iao_active.T, s1, iao_active))))

    return iao_active


def _iao_complementary_orbitals(mol, iao_ref):
    """Get the IAOs for complementary space (virtual orbitals).

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        iao_ref (numpy.array): IAO in occupied space (float64).

    Returns:
        numpy.array: IAO in complementary space (float64).
    """

    #   Get the total number of AOs
    norbital_total = mol.nao_nr()

    #   Calculate the Overlaps for total basis
    s1 = mol.intor_symmetric("int1e_ovlp")

    #   Construct the complementary space AO
    number_iaos = iao_ref.shape[1]
    number_inactive = norbital_total - number_iaos
    iao_com_ref = _iao_complementary_space(iao_ref, s1, number_inactive)

    #   Get a list of active orbitals
    min_mol = iao.reference_mol(mol)
    norbital_active, active_list = _iao_count_active(mol, min_mol)

    #   Obtain the Overlap-like matrices
    s21 = s1[active_list, :]
    s2 = s21[:, active_list]
    s12 = s21.T

    #   Calculate P_12 = S_1^-1 * S_12 using Cholesky decomposition
    s1_sqrt = scipy.linalg.cho_factor(s1)
    s2_sqrt = scipy.linalg.cho_factor(s2)
    p12 = scipy.linalg.cho_solve(s1_sqrt, s12)

    #   C~ = orth ( second_half ( S_1^-1 * S_12 * first_half ( S_2^-1 * S_21 * C ) ) )
    c_tilde = scipy.linalg.cho_solve(s2_sqrt, np.dot(s21, iao_com_ref))
    c_tilde = scipy.linalg.cho_solve(s1_sqrt, np.dot(s12, c_tilde))
    c_tilde = np.dot(c_tilde, orth.lowdin(reduce(np.dot, (c_tilde.T, s1, c_tilde))))

    #   Obtain C * C^T * S1 and C~ * C~^T * S1
    ccs1 = reduce(np.dot, (iao_com_ref, iao_com_ref.conj().T, s1))
    ctcts1 = reduce(np.dot, (c_tilde, c_tilde.conj().T, s1))

    #   Calculate A = ccs1 * ctcts1 * p12 + ( 1 - ccs1 ) * ( 1 - ctcts1 ) * p12
    iao_comp = (p12 + reduce(np.dot, (ccs1, ctcts1, p12)) * 2 - np.dot(ccs1, p12) - np.dot(ctcts1, p12))
    iao_comp = np.dot(iao_comp, orth.lowdin(reduce(np.dot, (iao_comp.T, s1, iao_comp))))

    return iao_comp


def _iao_count_active(mol, min_mol):
    """Figure out the basis functions matching with MINAO.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        min_mol (numpy.array): The molecule to simulate in minao basis.

    Returns:
        int: Number of active orbitals.
        list: List of active orbitals (int).
    """

    #   Initialize the list
    active_number_list = []

    #   Loop over all basis and see if there are labels matching with the MINAO ones
    for idx, total_basis in enumerate(mol.spheric_labels()):
        if all([min_basis != total_basis for min_basis in min_mol.spheric_labels()]):
            active_number_list.append(idx)

    #   Make the list a numpy array
    number_active = len(active_number_list)
    active_number_list = np.array(active_number_list)

    return number_active, active_number_list


def _iao_complementary_space(iao_ref, s, number_inactive):
    """Determine the complementary space orbitals.

    Args:
        iao_ref (numpy.array): IAO in occupied space.
        s (numpy.array): The overlap matrix.
        number_inactive (int): The number of inactive orbitals.

    Returns:
        numpy.array: The inactive part in IAO (float64).
    """

    #   Construct the "density matrix" for active space
    density_active = np.dot(iao_ref, iao_ref.T)

    #   Get the MO Coefficient from the IAO density matrix
    a_mat = reduce(np.dot, (s, density_active, s))
    eigval, eigvec = scipy.linalg.eigh(a=a_mat, b=s)

    #   Extract inactive part of "MO Coefficient" and return it
    eigen_vectors = eigvec[:, : number_inactive]

    return eigen_vectors


def _iao_atoms(mol, iao1, iao2):
    """Assign IAO to atom centers and rearrange the IAOs.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        mf (pyscf.scf.RHF): The mean field of the molecule.
        iao1 (numpy.array): IAO for occupied space (float64).
        iao2 (numpy.array): IAO for complementary space (float64).

    Returns:
        numpy.array: The rearranged IAO (float64).
    """

    # Calclate the integrals for assignment
    number_orbitals = mol.nao_nr()
    r_int1e = mol.intor("cint1e_r_sph", 3)
    iao_combine = np.hstack((iao1, iao2))

    # Calculate atom center for each orbital
    x = np.diag(reduce(np.dot, (iao_combine.T, r_int1e[0], iao_combine)))
    y = np.diag(reduce(np.dot, (iao_combine.T, r_int1e[1], iao_combine)))
    z = np.diag(reduce(np.dot, (iao_combine.T, r_int1e[2], iao_combine)))

    # Align the coordinates
    orbitals_temp = np.vstack((x, y, z))
    orbitals = orbitals_temp.T

    # Assign each orbital to atom center
    atom_list = _dmet_atom_list(mol, orbitals)

    # Prepare the orbital labels
    orb_list = _dmet_orb_list(mol, atom_list)

    # Rearrange the orbitals
    iao_combine = iao_combine[:, orb_list]

    # Orthogonalize the orbitals
    s1 = mol.intor_symmetric("int1e_ovlp")
    iao_combine = np.dot(iao_combine, orth.lowdin(reduce(np.dot, (iao_combine.T, s1, iao_combine))))

    return iao_combine


def _dmet_atom_list(mol, orbitals):
    """Assign IAO to atom centers and rearrange the IAOs.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        orbitals (numpy.array): Coordinates for the orbital centers (float64).

    Returns:
        list: The list for atom assignment for IAO (int).
    """

    # Initialize the list
    number_orbitals = mol.nao_nr()
    newlist = []

    # Calculate the distance from atom centers and determine the nearest
    for i in range(number_orbitals):
        i_temp = 0
        distance_temp = scipy.linalg.norm(orbitals[i, :] - mol.atom_coord(0))
        for j in range(1, mol.natm):
            distance = scipy.linalg.norm(orbitals[i, :] - mol.atom_coord(j))
            if (distance < distance_temp):
                distance_temp = distance
                i_temp = j
            else:
                pass
        newlist.append(i_temp)

    return newlist


def _dmet_orb_list(mol, atom_list):
    """Rearrange the orbital labels.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        atom_list (list): Atom list for IAO assignment (int).

    Returns:
        list: The orbital list in new order (int).
    """
    newlist = []
    for i in range(mol.natm):
        for j in range(mol.nao_nr()):
            if (atom_list[j] == i):
                newlist.append(j)

    return newlist
