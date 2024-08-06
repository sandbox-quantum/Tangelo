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

"""This module provides the QRE class for handling quantum resource estimation
(QRE) in quantum chemistry problems. The QRE class is initialized with a
`sec_mol` object, which is used to obtain one-body and two-body integrals,
relevant for QRE.
"""


class QRE:
    """A class for estimating quantum resources for a given molecule.

    Attributes:
        sec_mol (SecondQuantizedMolecule): Self-explanatory.
        one_body_int (array): The one-body integrals of the molecule.
        two_body_int (array): The two-body integrals of the molecule.

    Methods:
        benchq(threshold, **kwargs):
            Calculates the Toffoli and qubit cost using the benchq library.
        pennylane(**kwargs):
            Calculates the double factorization resource cost using the
            Pennylane library.
    """

    def __init__(self, sec_mol):
        """Initialize the QRE object with a given quantum molecule.

        Args:
            sec_mol (SecondQuantizedMolecule): Self-explanatory.
        """
        self.sec_mol = sec_mol
        _, self.one_body_int, self.two_body_int = self.sec_mol.get_integrals(fold_frozen=True)

    def benchq(self, threshold, **kwargs):
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

        return get_double_factorized_qpe_toffoli_and_qubit_cost(self.one_body_int,
            self.two_body_int, threshold, **kwargs)

    def pennylane(self, **kwargs):
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

        return DoubleFactorization(self.one_body_int, self.two_body_int, **kwargs)
