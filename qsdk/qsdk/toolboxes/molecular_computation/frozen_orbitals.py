"""
This module defines functions to get suggestions for freezing orbitals. Those functions take a pyscf.gto object
and return an integer or a list of orbital indices for freezing orbitals.
"""


def get_frozen_core(molecule):
    """Function to compute the number of frozen orbitals. This function is only
    for the core (occupied orbitals).

    Args:
        molecule (pyscf.gto): Molecule to be evaluated.

    Returns:
        frozen_core (int): First N molecular orbitals to freeze.
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
    frozen_core = sum([v * core_orbitals[k] for k, v in elements.items()])

    return frozen_core


def get_orbitals_excluding_homo_lumo(molecule, homo_minus_n=0, lumo_plus_n=0):
    """Function that returns a list of orbitals to freeze if the user wants to consider
    only a subset from HOMO(-homo_min_n) to LUMO(+lumo_plus_n) orbitals. Users
    should be aware of degeneracies, as this function does not take this property
    into account. Also, it is only relevant for closed-shell systems.

    Args:
        molecule (pyscf.gto): Molecule to be evaluated.
        homo_minus_n (int): Starting point at HOMO - homo_minus_n.
        lumo_plus_n (int): Ending point at LUMO + lumo_plus_n.

    Returns:
        frozen_orbitals (list of int): Frozen orbitals not detected in the active space.
    """

    # Getting the number of molecular orbitals. It also works with different
    # basis sets.
    n_molecular_orb = molecule.nao_nr()
    n_electrons = molecule.nelectron

    # Identify the HOMO and LUMO.
    n_homo = n_electrons // 2 - 1
    n_lumo = n_homo + 1

    frozen_orbitals = [n for n in range(n_molecular_orb) if n not in range(n_homo-homo_minus_n, n_lumo+lumo_plus_n+1)]

    return frozen_orbitals
