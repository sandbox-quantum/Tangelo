"""This module provides a common point for mapping operators and
 statevectors from Fermionic to Qubit encodings via any of:
 - Jordan-Wigner
 - Bravyi-Kitaev (Fenwick Tree implementation)
 - symmetry-conserving Bravyi-Kitaev (2-qubit reduction via Z2 taper)
"""
from enum import Enum

from qsdk.toolboxes.qubit_mappings.jordan_wigner import jordan_wigner
from qsdk.toolboxes.qubit_mappings.bravyi_kitaev import bravyi_kitaev
# from symmetry_conserving_bravyi_kitaev import symmetry_conserving_bravyi_kitaev

from openfermion import FermionOperator

class MappingType(Enum):
    """Enumeration of the mappings supported for qubit transformations."""
    JORDAN_WIGNER = 0
    BRAVYI_KITAEV = 1
    # SYMMETRY_CONSERVING_BK = 2 #TODO: test and implement scBK


def get_mapping(mapping):
    """Check mapping input selection, cast to integer type. User can select from
    Jordan-Wigner, Bravyi-Kitaev, or symmetry conserving Bravyi-Kitaev.
    TODO: scBK implementation
    Args:
        mapping (int or string): if int, options are: 
            Jordan Wigner (0), Bravyi Kitaev (1), and sc-BK (2).
            Otherwise, if string, options are:
            jordan_wigner / jw, bravyi_kitaev / bk, and symmetry_conserving_bravyi_kitaev / scbk

    Returns:
        mapping (int): integer from 0-2 inclusive.
    """
    if type(mapping) is int:
        if mapping > 2:
            raise ValueError('Invalid mapping choice. See get_mapping documentation for options.')
    elif type(mapping) is str:
        if mapping.lower() == 'jordan_wigner' or mapping.lower() == 'jw':
            mapping = 0
        elif mapping.lower() == 'bravyi_kitaev' or mapping.lower() == 'bk':
            mapping = 1
        #TODO: test and implement scBK
        # elif mapping.lower() == 'symmetry_conserving_bravyi_kitaev' or mapping.lower() == 'scbk':
        #     mapping = 2
        else:
            raise ValueError('Invalid mapping name. See get_mapping documentation for options.')

    return mapping


def fermion_to_qubit_mapping(fermion_operator, mapping, n_qubits = None, n_electrons = None):
    """Perform mapping of fermionic operator to qubit operator. This function is mostly a wrapper
    around standard openfermion code, with some important distinctions. We strictly enforce the
    specification of n_qubits for Bravyi-Kitaev type transformations, and n_electrons for scBK.
    In the absence of this information, these methods can return unexpected results if the input
    operator does not specifically address the highest-orbital/qubit index.
    TODO: finish implementation of scBK

    Args:
        fermion_operator (FermionOperator): operator to translate to qubit representation
        mapping (int or string): mapping selection, see get_mapping for options.
        n_qubits (int): number of qubits for destination operator. Not required for Jordan-Wigner
        n_electrons (int): number of occupied electron modes in problem. Required for symmetry
            conserving Bravyi-Kitaev only. 

    Returns:
        qubit_operator (QubitOperator): input operator, encoded in the qubit space.
    """    
    if not type(fermion_operator) is FermionOperator:
        raise TypeError("Invalid operator format. Must use openfermion FermionOperator.")
    
    mapping = get_mapping(mapping)

    if mapping == MappingType.JORDAN_WIGNER.value:
        qubit_operator = jordan_wigner(fermion_operator)
    elif mapping == MappingType.BRAVYI_KITAEV.value:
        if n_qubits is None:
            raise ValueError("Bravyi Kitaev requires specification of number of qubits.")
        qubit_operator = bravyi_kitaev(fermion_operator, n_qubits = n_qubits)
    #START OF SC-BK
    # elif mapping == MappingType.SYMMETRY_CONSERVING_BK.value:
    #     if n_qubits is None:
    #         raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of qubits.")
    #     if n_electrons is None:
    #         raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of electrons.")
    #     qubit_operator = symmetry_conserving_bravyi_kitaev(fermion_operator, n_qubits = n_qubits, n_electrons = n_electrons)
    else:
        raise ValueError('Requested mapping type not supported.')
    
    return qubit_operator
