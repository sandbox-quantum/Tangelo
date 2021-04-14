import unittest
from pyscf import gto
from openfermion.transforms import get_fermion_operator, jordan_wigner, reorder
from openfermion.utils import up_then_down

from agnostic_simulator import Simulator

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.ansatz_generator.rucc import RUCC


class UCCSDTest(unittest.TestCase):

    def test_rucc_wrong_n_params(self):
        """ Verify RUCC wrong number of parameters. """

        with self.assertRaises(ValueError):
            RUCC(n_var_params=999)
        
        with self.assertRaises(ValueError):
            RUCC(n_var_params="3")
        
        with self.assertRaises(ValueError):
            RUCC(n_var_params=3.141516)

        with self.assertRaises(AssertionError):
            ucc3 = RUCC(n_var_params=3)
            ucc3.build_circuit()
            ucc3.update_var_params([3.1415])


if __name__ == "__main__":
    unittest.main()
