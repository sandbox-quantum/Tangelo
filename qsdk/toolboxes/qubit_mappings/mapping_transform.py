"""This module provides a common point for mapping operators and
 statevectors from Fermionic to Qubit encodings via any of:
 - Jordan-Wigner
 - Bravyi-Kitaev (Fenwick Tree implementation)
 - symmetry-conserving Bravyi-Kitaev (2-qubit reduction via Z2 taper)
"""
from qsdk.toolboxes.operators import FermionOperator
from qsdk.toolboxes.qubit_mappings import jordan_wigner, bravyi_kitaev, symmetry_conserving_bravyi_kitaev


def fermion_to_qubit_mapping(fermion_operator, mapping, n_qubits=None, n_electrons=None):
    """Perform mapping of fermionic operator to qubit operator. This function is mostly a wrapper
    around standard openfermion code, with some important distinctions. We strictly enforce the
    specification of n_qubits for Bravyi-Kitaev type transformations, and n_electrons for scBK.
    In the absence of this information, these methods can return unexpected results if the input
    operator does not specifically address the highest-orbital/qubit index.

    Args:
        fermion_operator (FermionOperator): operator to translate to qubit representation
        mapping (string): options are-- 
           'JW' (Jordan Wigner), 'BK' (Bravyi Kitaev), 'scBK' (symmetry-conserving Bravyi Kitaev).
        n_qubits (int): number of qubits for destination operator. Not required for Jordan-Wigner
        n_electrons (int): number of occupied electron modes in problem. Required for symmetry
            conserving Bravyi-Kitaev only. 

    Returns:
        qubit_operator (QubitOperator): input operator, encoded in the qubit space.
    """
    if not type(fermion_operator) is FermionOperator:
        raise TypeError("Invalid operator format. Must use FermionOperator.")

    if mapping.upper() == 'JW':
        qubit_operator = jordan_wigner(fermion_operator)
    elif mapping.upper() == 'BK':
        if n_qubits is None:
            raise ValueError("Bravyi Kitaev requires specification of number of qubits.")
        qubit_operator = bravyi_kitaev(fermion_operator, n_qubits=n_qubits)
    elif mapping.upper() == 'SCBK':
        if n_qubits is None:
            raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of qubits.")
        if n_electrons is None:
            raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of electrons.")
        qubit_operator = symmetry_conserving_bravyi_kitaev(fermion_operator, n_qubits=n_qubits, n_electrons=n_electrons)
    else:
        raise ValueError('Requested mapping type not supported.')

    return qubit_operator
