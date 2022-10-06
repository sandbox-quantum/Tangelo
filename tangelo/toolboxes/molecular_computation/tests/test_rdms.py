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

import os
import unittest
import json
import numpy as np

from openfermion.utils import load_operator

from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.molecular_computation.rdms import energy_from_rdms, compute_rdms

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

ferm_op_of = load_operator("H2_ferm_op.data", data_directory=pwd_this_test + "/data", plain_text=True)
ferm_op = FermionOperator()
ferm_op.__dict__ = ferm_op_of.__dict__
ferm_op.n_spinorbitals = 4
ferm_op.n_electrons = 2
ferm_op.spin = 0

cs_data = json.load(open("./data/H2_cs_data.dat", "r"))

rdm1 = [[1.97453997e+00, -7.05987336e-17],
        [-7.05987336e-17, 2.54600303e-02]]

rdm2 = [
    [[[1.97453997e+00, -7.96423130e-17], [-7.96423130e-17, 3.21234218e-33]],
     [[-7.96423130e-17, -2.24213843e-01], [0.00000000e+00, 9.04357944e-18]]],
    [[[-7.96423130e-17, 0.00000000e+00], [-2.24213843e-01, 9.04357944e-18]],
     [[3.21234218e-33, 9.04357944e-18], [9.04357944e-18, 2.54600303e-02]]]
]

class RDMsUtilitiesTest(unittest.TestCase):

    def test_energy_from_rdms(self):
        """Same test as in test_molecule.SecondQuantizedMoleculeTest.test_energy_from_rdms,
        but from a fermionic operator instead.
        """
        e_rdms = energy_from_rdms(ferm_op, rdm1, rdm2)
        self.assertAlmostEqual(e_rdms, -1.1372701, delta=1e-5)

    def test_compute_rdms(self):
        """Load data and compute RDMs from frequency list"""
        rdm1r, rdm2r, rdm1ssr, rdm2ssr = compute_rdms(ferm_op, cs_data, "scbk", False)

        self.assertAlmostEqual((np.array(rdm1)-rdm1ssr).sum(), 0., delta=1e-5)
        #self.assertAlmostEqual(np.array(rdm2), rdm2ssr, delta=1e-5)
        #self.assertAlmostEqual(np.array(rdm1ss), rdm1ssr, delta=1e-5)
        #self.assertAlmostEqual(np.array(rdm2ss), rdm2ssr, delta=1e-5)

if __name__ == "__main__":
    unittest.main()
