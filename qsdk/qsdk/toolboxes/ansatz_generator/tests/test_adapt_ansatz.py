import unittest
import math

from qsdk.toolboxes.ansatz_generator.adapt_ansatz import ADAPTAnsatz
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.toolboxes.operators import QubitOperator, FermionOperator
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping

f_op = FermionOperator("2^ 3^ 0 1") - FermionOperator("0^ 1^ 2 3")
qu_op = fermion_to_qubit_mapping(f_op, "jw")
for key in qu_op.terms:
    qu_op.terms[key] = math.copysign(1., float(qu_op.terms[key].imag))


class ADAPTAnsatzTest(unittest.TestCase):

    def test_adaptansatz_init(self):
        """Verify behavior of ADAPTAnsatz class. """

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

    def test_adaptansatz_adding(self):
        """Verify operator addition behavior of ADAPTAnsatz class. """

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        self.assertEqual(ansatz.n_var_params, 1)
        self.assertEqual(ansatz.length_operators, [8])

    def test_adaptansatz_set_var_params(self):
        """Verify variational parameter tuning behavior of ADAPTAnsatz class. """

        ansatz = ADAPTAnsatz(n_spinorbitals=4, n_electrons=2)
        ansatz.build_circuit()

        ansatz.add_operator(qu_op)

        ansatz.set_var_params([1.999])
        self.assertEqual(ansatz.var_params, [1.999])

        with self.assertRaises(ValueError):
            ansatz.set_var_params([1.999, 2.999])


if __name__ == "__main__":
    unittest.main()
