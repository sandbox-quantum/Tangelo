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

import os
import unittest
import tempfile

from openfermion.utils import load_operator

from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.molecular_computation.fcidump import fermop_from_fcidump

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

ferm_op_of = load_operator("H2_ferm_op.data", data_directory=pwd_this_test + "/data", plain_text=True)
ferm_op_ref = FermionOperator()
ferm_op_ref.__dict__ = ferm_op_of.__dict__.copy()


class FCIDUMPUtilitiesTest(unittest.TestCase):

    def test_read_fcidump_to_fermionoperator(self):
        """Generating fermionic operator from a FCIDUMP file without passing
        through an IntegralSolver.
        """

        ferm_op = fermop_from_fcidump(os.path.abspath(pwd_this_test + "/data/H2_sto3g.fcidump"))
        self.assertTrue(ferm_op_ref.isclose(ferm_op))

    def test_write_fcidump(self):
        """Writing a FCIDUMP file from a SecondQuantizedMolecule. The test
        checks if the fermionic operator can be generated from the newly
        created FCIDUMP file.
        """
        tmp = tempfile.NamedTemporaryFile()

        mol_H2_sto3g.write_fcidump(tmp.name)
        ferm_op = fermop_from_fcidump(tmp.name)
        self.assertTrue(ferm_op_ref.isclose(ferm_op))


if __name__ == "__main__":
    unittest.main()
