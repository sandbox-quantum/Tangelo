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

"""Tools to define a reference state under a specified qubit-mapping, and
translate into a Circuit.
"""


import numpy as np
import warnings

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.qubit_mappings.jkmn import jkmn_prep_vector

from openfermion.transforms import bravyi_kitaev_code

available_mappings = {"JW", "BK", "SCBK", "JKMN"}


def get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    """Get integer vector corresponding to Hartree Fock reference state.
    Reference state will occupy up to the n_electron-th molecular orbital.
    Depending on convention, basis is ordered alternating spin-up/spin-down
    (updown = False), or all up, then all down (updown = True).

    Args:
        n_spinorbitals (int): number of spin-orbitals in register.
        n_electrons (int): number of electrons in system.
        mapping (string): specify mapping, see mapping_transform.py for options
            "JW" (Jordan Wigner), or "BK" (Bravyi Kitaev), or "SCBK"
            (symmetry-conserving Bravyi Kitaev) or "JKMN"
            (Jiang Kalev Mruczkiewicz Neven)
        up_then_down (boolean): if True, all up, then all down, if False,
            alternating spin up/down.

    Returns:
        numpy array of int: binary integer array indicating occupation of each
            spin-orbital.
    """
    if mapping.upper() not in available_mappings:
        raise ValueError(f"Invalid mapping selection. Select from: {available_mappings}")

    vector = np.zeros(n_spinorbitals, dtype=int)
    if spin:
        # if n_electrons is odd, then spin is also odd
        n_alpha = n_electrons//2 + spin//2 + (n_electrons % 2)
        n_beta = n_electrons//2 - spin//2
        vector[0:2*n_alpha:2] = 1
        vector[1:2*n_beta+1:2] = 1
    else:
        vector[:n_electrons] = 1
    return get_mapped_vector(vector, mapping, up_then_down)


def do_bk_transform(vector):
    """Apply Bravyi-Kitaev transformation to fermion occupation vector.
    Currently, simple wrapper on openfermion tools.

    Args:
        vector (numpy array of int): fermion occupation vector.

    Returns:
        numpy array of int: qubit-encoded occupation vector.
    """
    mat = bravyi_kitaev_code(len(vector)).encoder.toarray()
    vector_bk = np.mod(np.dot(mat, vector), 2)
    return vector_bk


def do_scbk_transform(vector, n_spinorbitals):
    """Instantiate qubit vector for symmetry-conserving Bravyi-Kitaev
    transformation. Based on implementation by Yukio Kawashima in DMET project.

    Args:
        vector (numpy array of int): fermion occupation vector.
        n_spinorbitals (int): number of qubits in register.

    Returns:
        numpy array of int: qubit-encoded occupation vector.
    """
    vector_bk = do_bk_transform(vector)
    vector = np.delete(vector_bk, n_spinorbitals - 1)
    vector = np.delete(vector, n_spinorbitals//2 - 1)
    return vector


def do_jkmn_transform(vector):
    """Instantiate qubit vector for JKMN transformation.

    Args:
        vector (numpy array of int): fermion occupation vector.

    Returns:
        numpy array of int: qubit-encoded occupation vector.
    """
    return jkmn_prep_vector(vector)


def get_mapped_vector(vector, mapping, up_then_down=False):
    """Return vector to generate circuit for a given occupation vector and mapping

    Args:
        vector (array of int): fermion occupation vector with up_then_down=False ordering.
            Number of spin-orbitals is assumed by length of array.
        mapping (str): One of the supported qubit mappings
        up_then_down (bool): if True, all up, then all down, if False,
            alternating spin up/down.

    Returns:
        array: The vector that generates the mapping occupations"""

    if up_then_down:
        vector = np.concatenate((vector[::2], vector[1::2]))
    if mapping.upper() == "JW":
        return vector
    elif mapping.upper() == "BK":
        return do_bk_transform(vector)
    elif mapping.upper() == "SCBK":
        if not up_then_down:
            warnings.warn("Symmetry-conserving Bravyi-Kitaev enforces all spin-up followed by all spin-down ordering.", RuntimeWarning)
            vector = np.concatenate((vector[::2], vector[1::2]))
        return do_scbk_transform(vector, len(vector))
    elif mapping.upper() == "JKMN":
        return do_jkmn_transform(vector)


def vector_to_circuit(vector):
    """Translate occupation vector into a circuit. Each occupied state
    corresponds to an X-gate on the associated qubit index.

    Args:
        vector (numpy array of int): occupation vector.

    Returns:
        Circuit: instance of tangelo.linq Circuit class.
    """

    n_qubits = len(vector)
    circuit = Circuit(n_qubits=n_qubits)

    for index, occupation in enumerate(vector):
        if occupation:
            gate = Gate("X", target=index)
            circuit.add_gate(gate)

    return circuit


def get_reference_circuit(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    """Build the Hartree-Fock state preparation circuit for the designated
    mapping.

    Args:
        n_spinorbitals (int): number of qubits in register.
        n_electrons (int): number of electrons in system.
        mapping (string): specify mapping, see mapping_transform.py for options
            "JW" (Jordan Wigner), or "BK" (Bravyi Kitaev), or "SCBK"
            (symmetry-conserving Bravyi Kitaev).
        up_then_down (boolean): if True, all up, then all down, if False,
            alternating spin up/down.
        spin (int): 2*S = n_alpha - n_beta.

    Returns:
        Circuit: instance of tangelo.linq Circuit class.
    """
    vector = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=up_then_down, spin=spin)
    circuit = vector_to_circuit(vector)
    return circuit
