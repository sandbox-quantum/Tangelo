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

"""This module defines functions to get suggestions for freezing orbitals. Those
functions take a molecule and return an integer or a list of orbital indices for
freezing orbitals. Depending on the function, a Molecule or a
SecondQuantizedMolecule object can be used.
"""

from collections import Counter

import numpy as np


def get_frozen_core(molecule):
    """Function to compute the number of frozen orbitals. This function is only
    for the core (occupied orbitals).

    Args:
        molecule (SecondQuantizedMolecule): Molecule to be evaluated.

    Returns:
        int: First N molecular orbitals to freeze.
    """

    # Freezing core of each atom. "Row" refers to a periodic table row.
    # 1st row: None are frozen.
    # 2nd row: 1s is frozen.
    # 3rd row: 1s, 2s, 2px, 2py and 2pz are frozen.
    core_orbitals = {
        "H": 0, "He": 0,
        "Li": 1, "Be": 1, "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
        "Na": 5, "Mg": 5, "Al": 5, "Si": 5, "P": 5, "S": 5, "Cl": 5, "Ar": 5
    }

    # Counting how many of each element is in the molecule.
    elements = Counter([e[0] for e in molecule.xyz])
    frozen_core = sum([v * core_orbitals.get(k, 0) for k, v in elements.items()])

    return frozen_core


def get_orbitals_excluding_homo_lumo(molecule, homo_minus_n=0, lumo_plus_n=0):
    """Function that returns a list of orbitals to freeze if the user wants to
    consider only a subset from HOMO(-homo_min_n) to LUMO(+lumo_plus_n)
    orbitals. Users should be aware of degeneracies, as this function does not
    take this property into account.

    Args:
        molecule (SecondQuantizedMolecule): Molecule to be evaluated.
        homo_minus_n (int): Starting point at HOMO - homo_minus_n.
        lumo_plus_n (int): Ending point at LUMO + lumo_plus_n.

    Returns:
        list of int: Frozen orbitals not detected in the active space.
    """

    # Getting the number of molecular orbitals. It also works with different
    # basis sets.
    n_molecular_orb = molecule.n_mos

    n_lumo = molecule.mo_occ.tolist().index(0.)
    n_homo = n_lumo - 1

    frozen_orbitals = [n for n in range(n_molecular_orb) if n not in range(n_homo-homo_minus_n, n_lumo+lumo_plus_n+1)]

    return frozen_orbitals


def convert_frozen_orbitals(sec_mol, frozen_orbitals):
    """This function converts an int or a list of frozen_orbitals into four
    categories:
    - Active and occupied MOs;
    - Active and virtual MOs;
    - Frozen and occupied MOs;
    - Frozen and virtual MOs.
    Each of them are list with MOs indices (first one is 0). Note that they
    are MOs labelled, not spin-orbitals (MOs * 2) indices.

    Args:
        sec_mol (SecondQuantizedMolecule): Self-explanatory
        frozen_orbitals (int or list of int): Number of MOs or MOs indices
            to freeze.

    Returns:
        tuple of list: Active occupied, frozen occupied, active virtual and
            frozen virtual orbital indices.
    """

    if frozen_orbitals == "frozen_core":
        frozen_orbitals = get_frozen_core(sec_mol) if not sec_mol.ecp else 0
    elif frozen_orbitals is None:
        frozen_orbitals = 0

    # First case: frozen_orbitals is an int.
    # The first n MOs are frozen.
    if isinstance(frozen_orbitals, (int, np.integer)):
        frozen_orbitals = list(range(frozen_orbitals))
        if sec_mol.uhf:
            frozen_orbitals = [frozen_orbitals, frozen_orbitals]
    # Second case: frozen_orbitals is a list of int.
    # All MOs with indices in this list are frozen (first MO is 0, second is 1, ...).
    # Everything else raise an exception.
    elif isinstance(frozen_orbitals, list):
        if sec_mol.uhf and not (len(frozen_orbitals) == 2 and
                                all(isinstance(_, (int, np.integer)) for _ in frozen_orbitals[0]) and
                                all(isinstance(_, (int, np.integer)) for _ in frozen_orbitals[1])):
            raise TypeError("frozen_orbitals argument must be a list of int for both alpha and beta electrons")
        elif not sec_mol.uhf and not all(isinstance(_, int) for _ in frozen_orbitals):
            raise TypeError("frozen_orbitals argument must be an (or a list of) integer(s).")
    else:
        raise TypeError("frozen_orbitals argument must be an (or a list of) integer(s)")

    if sec_mol.uhf:
        occupied, virtual = list(), list()
        frozen_occupied, frozen_virtual = list(), list()
        active_occupied, active_virtual = list(), list()
        n_active_electrons = list()
        n_active_mos = list()
        for e in range(2):
            occupied.append([i for i in range(sec_mol.n_mos) if sec_mol.mo_occ[e][i] > 0.])
            virtual.append([i for i in range(sec_mol.n_mos) if sec_mol.mo_occ[e][i] == 0.])

            frozen_occupied.append([i for i in frozen_orbitals[e] if i in occupied[e]])
            frozen_virtual.append([i for i in frozen_orbitals[e] if i in virtual[e]])

            # Redefined active orbitals based on frozen ones.
            active_occupied.append([i for i in occupied[e] if i not in frozen_occupied[e]])
            active_virtual.append([i for i in virtual[e] if i not in frozen_virtual[e]])

            # Calculate number of active electrons and active_mos
            n_active_electrons.append(round(sum([sec_mol.mo_occ[e][i] for i in active_occupied[e]])))
            n_active_mos.append(len(active_occupied[e] + active_virtual[e]))

        if n_active_electrons[0] + n_active_electrons[1] == 0:
            raise ValueError("There are no active electrons.")
        if (n_active_electrons[0] == 2*n_active_mos[0]) and (n_active_electrons[1] == 2*n_active_mos[1]):
            raise ValueError("All active orbitals are fully occupied.")
    else:
        occupied = [i for i in range(sec_mol.n_mos) if sec_mol.mo_occ[i] > 0.]
        virtual = [i for i in range(sec_mol.n_mos) if sec_mol.mo_occ[i] == 0.]

        frozen_occupied = [i for i in frozen_orbitals if i in occupied]
        frozen_virtual = [i for i in frozen_orbitals if i in virtual]

        # Redefined active orbitals based on frozen ones.
        active_occupied = [i for i in occupied if i not in frozen_occupied]
        active_virtual = [i for i in virtual if i not in frozen_virtual]

        # Calculate number of active electrons and active_mos
        n_active_electrons = round(sum([sec_mol.mo_occ[i] for i in active_occupied]))
        n_active_mos = len(active_occupied + active_virtual)

        # Exception raised here if there is no active electron.
        # An exception is raised also if all active orbitals are fully occupied.
        if n_active_electrons == 0:
            raise ValueError("There are no active electrons.")
        if n_active_electrons == 2*n_active_mos:
            raise ValueError("All active orbitals are fully occupied.")

    return active_occupied, frozen_occupied, active_virtual, frozen_virtual
