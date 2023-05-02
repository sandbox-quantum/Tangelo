# Copyright 2023 Good Chemistry Company.
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


"""Module that defines function to convert a fermionic operator to a boson
operator. For electronic system, it means to couple all the electrons by pair.
"""

import itertools

from tangelo.toolboxes.operators import BosonOperator, QubitOperator


def hard_core_boson_operator(ferm_op):
    """Function to extract the coefficient of the Hard-Core Bosonic (HCB)
    Hamiltonian.

    Refs:
        - V.E. Elfving, M. Millaruelo, J.A. Gámez, and C. Gogolin. Phys. Rev. A
            103, 032605 (2021).
        - N.T. Thang and P.T.T. Nga, Communications in Physics 21, 301 (2011).

    Args:
        ferm_op (FermionOperator): Self-explanatory.

    Returns:
        BosonOperator: Self-explanatory.
    """

    # Getting the molecular integrals.
    cte, e_sei, e_tei = ferm_op.get_coeffs(spatial=True)
    e_tei *= 2

    boson_op = BosonOperator((), cte)
    n_mos = e_sei.shape[0]
    for i, j in itertools.product(range(n_mos), repeat=2):
        if i == j:
            coeff = 2*e_sei[i, i] + e_tei[i, i, i, i]
            boson_op += BosonOperator(f"{i}^ {i}", coeff)
        else:
            r1_coeff = e_tei[i, i, j, j]
            boson_op += BosonOperator(f"{i}^ {j}", r1_coeff)

            r2_coeff = 2*e_tei[i, j, j, i] - e_tei[i, j, i, j]
            boson_op += BosonOperator(f"{i}^ {i} {j}^ {j}", r2_coeff)

    return boson_op


def boson_to_qubit_mapping(bos_op):
    """Function to convert a Bosonic operator to a qubit operator. As qubits are
    bosons, the mapping is similar to the Jordan-Wigner mapping, but without the
    trailing Pauli-Z to account for anticommutation of the creation and
    annihilation operators.

    In short, every creation operator b^{\dagger} (resp. annihilation b) are
    mapped to X-iY strings (resp. X+iY), where X and Y are referring to the
    Pauli matrices.

    Args:
        bos_op (BosonOperator): Self-explanatory.

    Returns:
        QubitOperator: Self-explanatory.
    """

    def b(p, dagger=False):
        prefactor = -1 if dagger else 1
        return QubitOperator(f"X{p}", 0.5) + QubitOperator(f"Y{p}", prefactor*0.5j)

    qu_op = QubitOperator((), bos_op.constant)
    for term, coeff in bos_op.terms.items():
        if not term:
            continue

        qubit_term = QubitOperator((), coeff)
        for mo, dagger in term:
            qubit_term *= b(mo, dagger)

        qu_op += qubit_term

    return qu_op
