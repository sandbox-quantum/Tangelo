# Copyright SandboxAQ 2021-2024.
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

"""This module provides functions for handling quantum resource estimation
(QRE) in quantum chemistry problems. The QRE functions take a
SecondQuantizedMolecule object, which is used to obtain one-body and two-body
integrals, relevant for QRE.
"""


def qre_benchq(sec_mol, threshold, **kwargs):
    """Calculate the Toffoli and qubit cost using the benchq library. For
    more details, see the benchq documentation:
    https://github.com/zapatacomputing/benchq/blob/main/src/benchq/problem_embeddings/qpe.py#L128

    Dependencies:
        - benchq
        - openfermionpycf

    Args:
        threshold (float): The threshold parameter for the double
            factorization algorithm.
        **kwargs: Additional parameters to pass to the `benchq` function.

    Returns:
        (int, int): A tuple containing the Toffoli and qubit cost.
    """
    from benchq.problem_embeddings.qpe import get_double_factorized_qpe_toffoli_and_qubit_cost

    _, one_body_int, two_body_int = sec_mol.get_integrals(fold_frozen=True)

    return get_double_factorized_qpe_toffoli_and_qubit_cost(one_body_int,
        two_body_int, threshold, **kwargs)


def qre_pennylane(sec_mol, **kwargs):
    """Calculate the double factorization resource cost using the Pennylane
    library. For more details, see the Pennylane documentation:
    https://docs.pennylane.ai/en/stable/code/api/pennylane.resource.DoubleFactorization.html

    Dependency:
        - pennylane

    Args:
        **kwargs: Additional parameters to pass to the
            `DoubleFactorization` constructor.

    Returns:
        DoubleFactorization: An instance of the Pennylane
            DoubleFactorization resource.
    """
    from pennylane.resource import DoubleFactorization

    _, one_body_int, two_body_int = sec_mol.get_integrals(fold_frozen=True)

    return DoubleFactorization(one_body_int, two_body_int, **kwargs)
