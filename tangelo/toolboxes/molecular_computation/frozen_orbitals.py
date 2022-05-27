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

"""This module defines functions to get suggestions for freezing orbitals. Those
functions take a molecule and return an integer or a list of orbital indexes for
freezing orbitals. Depending on the function, a Molecule or a
SecondQuantizedMolecule object can be used.
"""


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
    elements = {i: molecule.elements.count(i) for i in molecule.elements}
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
