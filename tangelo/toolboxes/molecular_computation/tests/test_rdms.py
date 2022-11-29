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
from numpy.testing import assert_allclose
from openfermion.utils import load_operator

from tangelo.toolboxes.measurements import RandomizedClassicalShadow
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.molecular_computation.rdms import energy_from_rdms, compute_rdms

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

ferm_op_of = load_operator("H2_ferm_op.data", data_directory=pwd_this_test + "/data", plain_text=True)
ferm_op = FermionOperator()
ferm_op.__dict__ = ferm_op_of.__dict__.copy()
ferm_op.n_spinorbitals = 4
ferm_op.n_electrons = 2
ferm_op.spin = 0

exp_data = json.load(open(pwd_this_test + "/data/H2_raw_exact.dat", "r"))

rdm1ssr = np.array([[1.97454854+0.j, 0.+0.j],
                 [0.+0.j, 0.02545146+0.j]])

rdm2ssr = np.array(
    [[[[ 1.97454853e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
       [ 0.00000000e+00+0.00000000e+00j,  5.92100152e-09+0.00000000e+00j]],
      [[ 0.00000000e+00+0.00000000e+00j, -2.24176575e-01+2.77555756e-17j],
       [ 5.92100077e-09+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j]]],
     [[[ 0.00000000e+00+0.00000000e+00j,  5.92100077e-09+0.00000000e+00j],
       [-2.24176575e-01-2.77555756e-17j,  0.00000000e+00+0.00000000e+00j]],
      [[ 5.92100152e-09+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
       [ 0.00000000e+00+0.00000000e+00j,  2.54514569e-02+0.00000000e+00j]]]])


class RDMsUtilitiesTest(unittest.TestCase):

    def test_energy_from_rdms(self):
        """Compute energy using known spin-summed 1RDM and 2RDM"""
        e_rdms = energy_from_rdms(ferm_op, rdm1ssr, rdm2ssr)
        self.assertAlmostEqual(e_rdms, -1.1372701, delta=1e-5)

    def test_compute_rdms_from_raw_data(self):
        """Compute RDMs from frequency list"""
        rdm1, rdm2, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, exp_data=exp_data)

        assert_allclose(rdm1ssr, rdm1ss, rtol=1e-5)
        assert_allclose(rdm2ssr, rdm2ss, rtol=1e-5)

    def test_compute_rdms_from_classical_shadow(self):
        """Compute RDMs from classical shadow"""
        # Construct ClassicalShadow
        bitstrings = []
        unitaries = []

        for b, hist in exp_data.items():
            for s, f in hist.items():
                factor = round(f * 10000)
                bitstrings.extend([s] * factor)
                unitaries.extend([b] * factor)

        cs_data = RandomizedClassicalShadow(unitaries=unitaries, bitstrings=bitstrings)

        rdm1, rdm2, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, shadow=cs_data, k=5)

        # Have to adjust tolerance to account for classical shadow rounding to 10000 shots
        assert_allclose(rdm1ssr, rdm1ss, atol=0.05)
        assert_allclose(rdm2ssr, rdm2ss, atol=0.05)


if __name__ == "__main__":
    unittest.main()
