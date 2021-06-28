""" This module defines the penatly terms that can be added to the fermionic Hamiltonian, 
    providing the ability to restrict the Hilbert space of solutions using VQE"""

from qsdk.toolboxes.ansatz_generator._general_unitary_cc import get_spin_ordered
from qsdk.toolboxes.operators import FermionOperator, normal_ordered


def number_operator_penalty(n_orbs, n_electrons, mu=1, up_then_down=False):
    R"""Function to generator the normal ordered number opeator penalty term as a FermionicOperator

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        n_electrons (int): number of electrons
        mu (float): Positive number in front of penalty term 
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        penalty_op (FermionicOperator): The number operator penalty term mu*(n_electrons-\hat{N})^2"""

    all_terms = list()
    all_terms.append([(), n_electrons])
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0)), -1])
        all_terms.append([((dn[0], 1), (dn[1], 0)), -1])

    penalty_op = FermionOperator()
    for item in all_terms:
        penalty_op += FermionOperator(item[0], item[1])
    penalty_op *= penalty_op
    penalty_op = normal_ordered(mu*penalty_op)
    return penalty_op


def spin_operator_penalty(n_orbs, sz, mu=1, up_then_down=False):
    R"""Function to generator the normal ordered Sz opeator penalty term

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        sz (int): the desired Sz quantum number to penalize for
        mu (float): Positive number in front of penalty term 
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        penalty_op (FermionicOperator): The Sz operator penalty term mu*(sz-\hat{Sz})^2"""

    all_terms = list()
    all_terms.append([(), sz])
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0)), -1/2])
        all_terms.append([((dn[0], 1), (dn[1], 0)), 1/2])

    penalty_op = FermionOperator()
    for item in all_terms:
        penalty_op += FermionOperator(item[0], item[1])
    penalty_op *= penalty_op
    penalty_op = normal_ordered(mu*penalty_op)
    return penalty_op


def spin2_operator_penalty(n_orbs, s2, mu=1, up_then_down=False):
    R"""Function to generator the normal ordered S^2 opeator penalty term

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        s2 (int): the desired S^2 quantum number to penalize for
        mu (float): Positive number in front of penalty term 
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        penalty_op (FermionicOperator): The S^2 operator penalty term mu*(s2-\hat{S}^2)^2"""

    all_terms = list()
    all_terms.append([(), s2])
    for i in range(n_orbs):
        up, dn = get_spin_ordered(n_orbs, i, i, up_down=up_then_down)  # get spin-orbital indices
        all_terms.append([((up[0], 1), (up[1], 0), (up[0], 1), (up[1], 0)), -1/4])
        all_terms.append([((dn[0], 1), (dn[1], 0), (dn[0], 1), (dn[1], 0)), -1/4])
        all_terms.append([((up[0], 1), (up[1], 0), (dn[0], 1), (dn[1], 0)), 1/4])
        all_terms.append([((dn[0], 1), (dn[1], 0), (up[0], 1), (up[1], 0)), 1/4])
        for j in range(n_orbs):
            if (i != j):
                up2, dn2 = get_spin_ordered(n_orbs, j, j, up_down=up_then_down)
                all_terms.append([((up[0], 1), (up[1], 0), (up2[0], 1), (up2[1], 0)), -1/4])
                all_terms.append([((dn[0], 1), (dn[1], 0), (dn2[0], 1), (dn2[1], 0)), -1/4])
                all_terms.append([((up[0], 1), (up[1], 0), (dn2[0], 1), (dn2[1], 0)), 1/4])
                all_terms.append([((dn[0], 1), (dn[1], 0), (up2[0], 1), (up2[1], 0)), 1/4])
                all_terms.append([((up[0], 1), (dn[1], 0), (dn2[0], 1), (up2[1], 0)), -1/2])
                all_terms.append([((dn[0], 1), (up[1], 0), (up2[0], 1), (dn2[1], 0)), -1/2])

    penalty_op = FermionOperator()
    for item in all_terms:
        penalty_op += FermionOperator(item[0], item[1])
    penalty_op *= penalty_op
    penalty_op = normal_ordered(mu*penalty_op)
    return penalty_op
