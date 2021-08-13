""" This module defines various kinds of operators used in vqe.
    It can later be broken down in several modules if needed. """

# Later on, if needed, we can extract the code for the operators themselves to remove the dependencies and customize
import openfermion


class FermionOperator(openfermion.FermionOperator):
    """ Currently, this class is coming from openfermion. Can be later on be replaced by our own implementation. """
    pass


class QubitOperator(openfermion.QubitOperator):
    """ Currently, this class is coming from openfermion. Can be later on be replaced by our own implementation. """
    pass


class QubitHamiltonian(QubitOperator):
    """Docstring. """
    pass


def count_qubits(qb_op):
    """ Return the number of qubits used by the qubit operator based on the highest index found in the terms."""
    if (len(qb_op.terms.keys()) == 0) or ((len(qb_op.terms.keys()) == 1) and (len(list(qb_op.terms.keys())[0]) == 0)):
        return 0
    else:
        return max([(sorted(pw))[-1][0] for pw in qb_op.terms.keys() if len(pw) > 0]) + 1


def normal_ordered(fe_op):
    """ Input: a Fermionic operator of class toolboxes.operators.FermionicOperator or openfermion.FermionicOperator
        Return: normal ordered toolboxes.operators.FermionicOperator"""

    # Obtain normal ordered fermionic operator as list of terms
    norm_ord_terms = openfermion.transforms.normal_ordered(fe_op).terms

    # Regeneratore full operator using class of qsdk.toolboxes.operators.FermionicOperator
    norm_ord_fe_op = FermionOperator()
    for term in norm_ord_terms:
        norm_ord_fe_op += FermionOperator(term, norm_ord_terms[term])
    return norm_ord_fe_op


def squared_normal_ordered(all_terms):
    """ Input: a list of terms to generate toolboxes.operators.FermionOperator or openfermion.FermionOperator
        Return: squared (i.e. fe_op*fe_op) and normal ordered toolboxes.operators.FermionOperator"""

    # Obtain normal ordered fermionic operator as list of terms
    fe_op = list_to_fermionoperator(all_terms)
    fe_op *= fe_op
    return normal_ordered(fe_op)


def list_to_fermionoperator(all_terms):
    """ Input: a list of terms to generate FermionOperator
        Return: a toolboxes.operators.FermionOperator"""

    fe_op = FermionOperator()
    for item in all_terms:
        fe_op += FermionOperator(item[0], item[1])
    return fe_op
