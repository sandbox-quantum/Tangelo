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

"""This module defines the penatly terms that can be added to the target
fermionic Hamiltonian, providing the ability to restrict the Hilbert space of
solutions using VQE. For example usages see
    - Illa G. Ryabinkin, Scott N. Genin, Artur F. Izmaylov. "Constrained
        variational quantum eigensolver: Quantum computer search engine in the
        Fock space" https://arxiv.org/abs/1806.00461.
    - Gabriel Greene-Diniz, David Munoz Ramo. "Generalized unitary coupled
        cluster excitations for multireference molecular states optimized by the
        Variational Quantum Eigensolver" https://arxiv.org/abs/1910.05168.
"""

from tangelo.toolboxes.ansatz_generator.fermionic_operators import number_operator_list, spinz_operator_list, spin2_operator_list
from tangelo.toolboxes.operators import FermionOperator, squared_normal_ordered


def number_operator_penalty(n_orbs, n_electrons, mu=1, up_then_down=False):
    r"""Function to generate the normal ordered number operator penalty term as
    a FermionOperator.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        n_electrons (int): number of electrons.
        mu (float): Positive number in front of penalty term.
        up_then_down (bool): The ordering of the spin orbitals.
            qiskit (True) openfermion (False)
            If later transforming to qubits, one should generally let the qubit
            mapping handle the ordering.

    Returns:
        FermionOperator: The number operator penalty term
            mu*(\hat{N}-n_electrons)^2.
    """

    all_terms = [[(), -n_electrons]] + number_operator_list(n_orbs, up_then_down)

    return mu*squared_normal_ordered(all_terms)


def spin_operator_penalty(n_orbs, sz, mu=1, up_then_down=False):
    r"""Function to generate the normal ordered Sz operator penalty term.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        sz (int): the desired Sz quantum number to penalize for.
        mu (float): Positive number in front of penalty term.
        up_then_down: The ordering of the spin orbitals.
            qiskit (True) openfermion (False)
            If later transforming to qubits, one should generally let the qubit
            mapping handle the ordering.

    Returns:
        FermionOperator: The Sz operator penalty term mu*(\hat{Sz}-sz)^2.
    """

    all_terms = [[(), -sz]] + spinz_operator_list(n_orbs, up_then_down)

    return mu*squared_normal_ordered(all_terms)


def spin2_operator_penalty(n_orbs, s2, mu=1, up_then_down=False):
    r"""Function to generate the normal ordered S^2 operator penalty term,
    operator form taken from
    https://pubs.rsc.org/en/content/articlepdf/2019/cp/c9cp02546d.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2).
        s2 (int): the desired S^2 quantum number to penalize for.
            singlet: s2=0*(0+1)=0, doublet: s2=(1/2)*(1/2+1)=3/4, triplet,
            s2=1*(1+1)=2 ...
        mu (float): Positive number in front of penalty term.
        up_then_down: The ordering of the spin orbitals.
            qiskit (True) openfermion (False)
            If later transforming to qubits, one should generally let the qubit
            mapping handle the ordering.

    Returns:
        FermionOperator: The S^2 operator penalty term mu*(\hat{S}^2-s2)^2.
    """

    all_terms = [[(), -s2]] + spin2_operator_list(n_orbs, up_then_down)

    return mu*squared_normal_ordered(all_terms)


def combined_penalty(n_orbs, opt_penalty_terms=None, up_then_down=False):
    r"""Function to generate the sum of all available penalty terms, currently
    implemented are
    - "N": number operator with eigenvalue (number of particles).
    - "Sz": Sz|s,m_s> = ms|s,m_s>.
    - "S^2": S^2|s,m_s> = s(s+1)|s,m_s>.

    Args:
        n_orbs (int): number of active orbitals in the fermion basis (this is
            number of spin-orbitals divided by 2).
        opt_penalty_terms (dict): The options for each penalty "N", "Sz", "S^2"
            as
            - "N" (array or list[float]):
                [Prefactor, Value] Prefactor * (\hat{N} - Value)^2
            - "Sz" (array or list[float]):
                [Prefactor, Value] Prefactor * (\hat{Sz} - Value)^2
            - "S^2" (array or list[float]):
                [Prefactor, Value] Prefactor * (\hat{S}^2 - Value)^2
        up_then_down: The ordering of the spin orbitals.
            qiskit (True) openfermion (False)
            If later transforming to qubits, one should generally let the qubit
            mapping handle this.

    Returns:
        FermionOperator: The combined n_electron+sz+s^2 penalty
            terms.
    """

    penalty_terms = {"N": [0, 0], "Sz": [0, 0], "S^2": [0, 0]}
    if opt_penalty_terms:
        for k, v in opt_penalty_terms.items():
            if k in penalty_terms:
                penalty_terms[k] = v
            else:
                raise KeyError(f"Keyword :: {k}, penalty term not available")
    else:
        return FermionOperator()

    pen_ferm = FermionOperator()
    if (penalty_terms["N"][0] > 0):
        prefactor, n_electrons = penalty_terms["N"][:]
        pen_ferm += number_operator_penalty(n_orbs, n_electrons, mu=prefactor, up_then_down=up_then_down)
    if (penalty_terms["Sz"][0] > 0):
        prefactor, sz = penalty_terms["Sz"][:]
        pen_ferm += spin_operator_penalty(n_orbs, sz, mu=prefactor, up_then_down=up_then_down)
    if (penalty_terms["S^2"][0] > 0):
        prefactor, s2 = penalty_terms["S^2"][:]
        pen_ferm += spin2_operator_penalty(n_orbs, s2, mu=prefactor, up_then_down=up_then_down)
    return pen_ferm
