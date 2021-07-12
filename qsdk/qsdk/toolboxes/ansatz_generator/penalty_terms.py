""" This module defines the penatly terms that can be added to the fermionic Hamiltonian,
    providing the ability to restrict the Hilbert space of solutions using VQE"""

from qsdk.toolboxes.ansatz_generator._general_unitary_cc import get_spin_ordered
from qsdk.toolboxes.operators import FermionOperator, normal_ordered


def number_operator_penalty(n_orbs, n_electrons, mu=1, up_then_down=False):
    r"""Function to generator the normal ordered number opeator penalty term as a FermionicOperator

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
    r"""Function to generator the normal ordered Sz opeator penalty term

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
    r"""Function to generator the normal ordered S^2 opeator penalty term, operator form taken from
        https://pubs.rsc.org/en/content/articlepdf/2019/cp/c9cp02546d

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
        all_terms.append([((up[0], 1), (dn[1], 0), (dn[0], 1), (up[1], 0)), -1/2])
        all_terms.append([((dn[0], 1), (up[1], 0), (up[0], 1), (dn[1], 0)), -1/2])
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


def penalty(n_orbs, opt_penalty_terms=None, up_then_down=False):
    r"""Function to generator the normal ordered combined N, Sz, and S^2 opeator penalty term

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        penalty_terms (dict): The options for each penalty 'n_electron', 'sz', 's2' as
                "n_electron" (array or list): [Prefactor, Value] Prefactor * (\hat{N} - Value)^2
                "sz" (list[float]): [Prefactor, Value] Prefactor * (\hat{Sz} - Value)^2
                "s2" (list[float]): [Prefactor, Value] Prefactor * (\hat{S}^2 - Value)^2
        up_then_down: The ordering of the spin orbitals. Should generally let the
                      qubit mapping handle this but can do it here as well.

    Returns:
        penalty_op (FermionicOperator): The combined n_electron+sz+s^2 penalty terms"""

    penalty_terms = {"n_electron": [0, 0], 'sz': [0, 0], 's2': [0, 0]}
    if opt_penalty_terms:
        for k, v in opt_penalty_terms.items():
            if k in penalty_terms:
                penalty_terms[k] = v
            else:
                raise KeyError(f"Keyword :: {k}, penalty term not available")
    else:
        return FermionOperator()

    pen_ferm = FermionOperator()
    if (penalty_terms["n_electron"][0] > 0):
        prefactor = penalty_terms["n_electron"][0]
        n_electrons = penalty_terms["n_electron"][1]
        pen_ferm += number_operator_penalty(n_orbs, n_electrons, mu=prefactor, up_then_down=up_then_down)
    if (penalty_terms["sz"][0] > 0):
        prefactor = penalty_terms["sz"][0]
        sz = penalty_terms["sz"][1]
        pen_ferm += spin_operator_penalty(n_orbs, sz, mu=prefactor, up_then_down=up_then_down)
    if (penalty_terms["s2"][0] > 0):
        prefactor = penalty_terms["s2"][0]
        s2 = penalty_terms["s2"][1]
        pen_ferm += spin2_operator_penalty(n_orbs, s2, mu=prefactor, up_then_down=up_then_down)
    return pen_ferm
