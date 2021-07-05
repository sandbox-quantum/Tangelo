R""" This module defines the fermionic operators that can be used to obtain expectation values
    of commonly used quantum numbers. The available operators are
    1) N: number of electrons
    2) Sz: The spin z-projection Sz|\psi>=m_s|\psi>
    3) S^2: The spin quantum number S^2|\psi>=s(s+1)|\psi> associated with spin angular momentum
    which allows one to decide whether the state has the correct properties."""

from qsdk.toolboxes.ansatz_generator._general_unitary_cc import get_spin_ordered
from qsdk.toolboxes.operators import FermionOperator, normal_ordered


def number_operator(n_orbs, up_then_down=False):
    R"""Function to generator the normal ordered number opeator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        num_op (FermionicOperator): The number operator penalty hat{N}"""

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0)), 1])
        all_terms.append([((dn[0], 1), (dn[1], 0)), 1])

    num_op = FermionOperator()
    for item in all_terms:
        num_op += FermionOperator(item[0], item[1])
    num_op = normal_ordered(num_op)
    return num_op


def spinz_operator(n_orbs, up_then_down=False):
    R"""Function to generator the normal ordered Sz opeator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        spin_op (FermionicOperator): The Sz operator \hat{Sz}"""

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0)), 1/2])
        all_terms.append([((dn[0], 1), (dn[1], 0)), -1/2])

    spinz_op = FermionOperator()
    for item in all_terms:
        spinz_op += FermionOperator(item[0], item[1])
    spinz_op = normal_ordered(spinz_op)
    return spinz_op


def spin2_operator(n_orbs, up_then_down=False):
    R"""Function to generator the normal ordered S^2 opeator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        spin2_op (FermionicOperator): The S^2 operator \hat{S}^2"""

    all_terms = list()
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0), (up[0], 1), (up[1], 0)), 1/4])
        all_terms.append([((dn[0], 1), (dn[1], 0), (dn[0], 1), (dn[1], 0)), 1/4])
        all_terms.append([((up[0], 1), (up[1], 0), (dn[0], 1), (dn[1], 0)), -1/4])
        all_terms.append([((dn[0], 1), (dn[1], 0), (up[0], 1), (up[1], 0)), -1/4])
        all_terms.append([((up[0], 1), (dn[1], 0), (dn[0], 1), (up[1], 0)), 1/2])
        all_terms.append([((dn[0], 1), (up[1], 0), (up[0], 1), (dn[1], 0)), 1/2])
        for j in range(n_orbs):
            if (i != j):
                up2, dn2 = get_spin_ordered(n_orbs, j, j, up_down=up_then_down)
                all_terms.append([((up[0], 1), (up[1], 0), (up2[0], 1), (up2[1], 0)), 1/4])
                all_terms.append([((dn[0], 1), (dn[1], 0), (dn2[0], 1), (dn2[1], 0)), 1/4])
                all_terms.append([((up[0], 1), (up[1], 0), (dn2[0], 1), (dn2[1], 0)), -1/4])
                all_terms.append([((dn[0], 1), (dn[1], 0), (up2[0], 1), (up2[1], 0)), -1/4])
                all_terms.append([((up[0], 1), (dn[1], 0), (dn2[0], 1), (up2[1], 0)), 1/2])
                all_terms.append([((dn[0], 1), (up[1], 0), (up2[0], 1), (dn2[1], 0)), 1/2])

    spin2_op = FermionOperator()
    for item in all_terms:
        spin2_op += FermionOperator(item[0], item[1])
    spin2_op = normal_ordered(spin2_op)
    return spin2_op
