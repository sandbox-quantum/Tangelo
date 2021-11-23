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

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Module to remove two qubits from the problem space using conservation of
electron number and conservation of electron spin. As described in
arXiv:1701.08213 and Phys. Rev. X 6, 031007.
"""


import numpy as np
import copy

from openfermion import FermionOperator as ofFermionOperator
from openfermion.transforms import bravyi_kitaev_tree, reorder
from openfermion.utils import count_qubits
from openfermion.utils import up_then_down as up_then_down_order


def symmetry_conserving_bravyi_kitaev(fermion_operator, n_spinorbitals,
                                      n_electrons, up_then_down=False, spin=0):
    """Returns the QubitOperator for the FermionOperator supplied, with two
    qubits removed using conservation of (parity) of electron spin and number,
    as described in arXiv:1701.08213.  This function has been modified from its
    analogous implementation in openfermion in order to circumvent failures when
    passing a fermion_operator which does not explicitly reference the highest
    index qubit in the register.

    Args:
        fermion_operator (FermionOperator): fermionic operator to transform to
            QubitOperator.
        n_spinorbitals (int): The number of active spin-orbitals being
            considered for the system.
        n_electrons (int): The number of active fermions being considered for
            the system (note, this is less than the number of electrons in a
            molecule if some orbitals have been frozen).
        up_then_down (bool): specify if the spin-orbital basis is already
            ordered putting all spin up before all spin down states.

    Returns:
        QubitOperator: The qubit operator corresponding to the supplied
            fermionic operator, with two qubits removed using spin symmetries.

    WARNING:
        Reorders orbitals from the default even-odd ordering to all spin-up
        orbitals, then all spin-down orbitals.

    Raises:
        ValueError if fermion_hamiltonian isn"t of the type FermionOperator, or
        active_orbitals isn"t an integer, or active_fermions isn"t an integer.

    Notes: This function reorders the spin orbitals as all spin-up, then all
        spin-down. It uses the OpenFermion bravyi_kitaev_tree mapping, rather
        than the bravyi-kitaev mapping. Caution advised when using with a
        Fermi-Hubbard Hamiltonian; this technique correctly reduces the
        Hamiltonian only for the lowest energy even and odd fermion number
        states, not states with an arbitrary number of fermions.
    """
    if not isinstance(fermion_operator, ofFermionOperator):
        raise ValueError("Supplied operator should be an instance "
                         "of openfermion FermionOperator class.")
    if type(n_spinorbitals) is not int:
        raise ValueError("Number of spin-orbitals should be an integer.")
    if type(n_electrons) is not int:
        raise ValueError("Number of electrons should be an integer.")
    if n_spinorbitals < count_qubits(fermion_operator):
        raise ValueError("Number of spin-orbitals is too small for FermionOperator input.")
    # Check that the input operator is suitable for application of scBK
    check_operator(fermion_operator, num_orbitals=(n_spinorbitals//2), up_then_down=up_then_down)

    # If necessary, arrange spins up then down, then BK map to qubit Hamiltonian.
    if not up_then_down:
        fermion_operator = reorder(fermion_operator, up_then_down_order, num_modes=n_spinorbitals)
    qubit_operator = bravyi_kitaev_tree(fermion_operator, n_qubits=n_spinorbitals)
    qubit_operator.compress()

    n_alpha = n_electrons//2 + spin//2 + (n_electrons % 2)

    # Allocates the parity factors for the orbitals as in arXiv:1704.08213.
    parity_final_orb = (-1)**n_electrons
    parity_middle_orb = (-1)**n_alpha

    # Removes the final qubit, then the middle qubit.
    qubit_operator = edit_operator_for_spin(qubit_operator,
                                            n_spinorbitals,
                                            parity_final_orb)
    qubit_operator = edit_operator_for_spin(qubit_operator,
                                            n_spinorbitals/2,
                                            parity_middle_orb)

    # We remove the N/2-th and N-th qubit from the register.
    to_prune = (n_spinorbitals//2 - 1, n_spinorbitals - 1)
    qubit_operator = prune_unused_indices(qubit_operator, prune_indices=to_prune, n_qubits=n_spinorbitals)

    return qubit_operator


def edit_operator_for_spin(qubit_operator, spin_orbital, orbital_parity):
    """Removes the Z terms acting on the orbital from the operator. For qubits
    to be tapered out, the action of Z-operators in operator terms are reduced
    to the associated eigenvalues. This simply corresponds to multiplying term
    coefficients by the related eigenvalue +/-1.

    Args:
        qubit_operator (QubitOperator): input operator.
        spin_orbital (int): index of qubit encoding (spin/occupation) parity.
        orbital_parity (int): plus/minus one, parity of eigenvalue.

    Returns:
        QubitOperator: updated operator, with relevant coefficients multiplied
            by +/-1.
    """
    new_qubit_dict = {}
    for term, coefficient in qubit_operator.terms.items():
        # If Z acts on the specified orbital, precompute its effect and
        # remove it from the Hamiltonian.
        if (spin_orbital - 1, "Z") in term:
            new_coefficient = coefficient*orbital_parity
            new_term = tuple(i for i in term if i != (spin_orbital - 1, "Z"))
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(new_term) is None:
                new_qubit_dict[new_term] = new_coefficient
            else:
                old_coefficient = new_qubit_dict.get(new_term)
                new_qubit_dict[new_term] = new_coefficient + old_coefficient
        else:
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(term) is None:
                new_qubit_dict[term] = coefficient
            else:
                old_coefficient = new_qubit_dict.get(term)
                new_qubit_dict[term] = coefficient + old_coefficient

    qubit_operator.terms = new_qubit_dict
    qubit_operator.compress()

    return qubit_operator


def prune_unused_indices(qubit_operator, prune_indices, n_qubits):
    """Rewritten from openfermion implementation. This uses the number of qubits,
    rather than the operator itself to specify the number of qubits relevant to
    the problem. This is especially important for, e.g. terms in the ansatz
    which may not individually pertain to all qubits in the problem.

    Remove indices that do not appear in any terms.

    Indices will be renumbered such that if an index i does not appear in any
    terms, then the next largest index that appears in at least one term will be
    renumbered to i.

    Args:
        qubit_operator (QubitOperator): input operator.
        prune_indices (tuple of int): indices to be removed from qubit register.
        n_qubits (int): number of qubits in register.

    Returns:
        QubitOperator: output operator, with designated qubit indices excised.
    """

    indices = np.linspace(0, n_qubits - 1, n_qubits, dtype=int)
    indices = np.delete(indices, prune_indices)

    # Construct a dict that maps the old indices to new ones
    index_map = {}
    for index in enumerate(indices):
        index_map[index[1]] = index[0]

    new_operator = copy.deepcopy(qubit_operator)
    new_operator.terms.clear()

    # Replace the indices in the terms with the new indices
    for term in qubit_operator.terms:
        new_term = [(index_map[op[0]], op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = qubit_operator.terms[term]

    return new_operator


def check_operator(fermion_operator, num_orbitals=None, up_then_down=False):
    """Check if the input fermion operator is suitable for application of
    symmetry-consering BK qubit reduction. Excitation must: preserve parity of
    fermion occupation, and parity of spin expectation value. This assumes
    alternating spin-up/spin-down ordering of input operator.

    Args:
        fermion_operator (FermionOperator): input fermionic operator.
        num_orbitals (int): specify number of orbitals (number of modes / 2),
            required for up then down ordering.
        up_then_down (bool): True if all spin up before all spin down, otherwise
            alternates.
    """
    if up_then_down and (num_orbitals is None):
        raise ValueError("Up then down spin ordering requires number of modes specified.")
    for term in fermion_operator.terms:
        number_change = 0
        spin_change = 0
        for index, action in term:
            number_change += 2*action - 1
            if up_then_down:
                spin_change += (2*action - 1)*(-2*(index // num_orbitals) + 1)*0.5

            else:
                spin_change += (2*action - 1)*(-2*(index % 2) + 1)*0.5
        if number_change % 2 != 0:
            raise ValueError("Invalid operator: input fermion operator does not conserve occupation parity.")
        if spin_change % 2 != 0:
            raise ValueError("Invalid operator: input fermion operator does not conserve spin parity.")
