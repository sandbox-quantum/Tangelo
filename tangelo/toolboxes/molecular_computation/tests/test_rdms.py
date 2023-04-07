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

import os
import unittest
import json

import numpy as np
from numpy.testing import assert_allclose
from openfermion.utils import load_operator

from tangelo import SecondQuantizedMolecule
from tangelo.molecule_library import mol_H2O_sto3g
from tangelo.toolboxes.measurements import RandomizedClassicalShadow
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.molecular_computation.rdms import energy_from_rdms, compute_rdms, \
     pad_rdms_with_frozen_orbitals_restricted, pad_rdms_with_frozen_orbitals_unrestricted
from tangelo.linq.helpers import pauli_string_to_of, pauli_of_to_string
from tangelo.toolboxes.post_processing import Histogram, aggregate_histograms

# For openfermion.load_operator function.
pwd_this_test = os.path.dirname(os.path.abspath(__file__))

ferm_op_of = load_operator("H2_ferm_op.data", data_directory=pwd_this_test + "/data", plain_text=True)
ferm_op = FermionOperator()
ferm_op.__dict__ = ferm_op_of.__dict__.copy()
ferm_op.n_spinorbitals = 4
ferm_op.n_electrons = 2
ferm_op.spin = 0

exp_data_strings = json.load(open(pwd_this_test + "/data/H2_raw_exact.dat", "r"))
exp_data_tuples = {pauli_string_to_of(k): v for k, v in exp_data_strings.items()}

temp_hist_tuples = {k: Histogram(freqs, 10000) for k, freqs in exp_data_tuples.items()}
temp_hist_tuples[((0, "Z"),)] = aggregate_histograms(temp_hist_tuples[((0, "Z"), (1, "Z"))], temp_hist_tuples[((0, "Z"), (1, "X"))])
temp_hist_tuples[((1, "Z"),)] = aggregate_histograms(temp_hist_tuples[((0, "Z"), (1, "Z"))], temp_hist_tuples[((0, "X"), (1, "Z"))])
temp_hist_tuples[((0, "X"),)] = aggregate_histograms(temp_hist_tuples[((0, "X"), (1, "Z"))], temp_hist_tuples[((0, "X"), (1, "X"))])
temp_hist_tuples[((1, "X"),)] = aggregate_histograms(temp_hist_tuples[((0, "Z"), (1, "X"))], temp_hist_tuples[((0, "X"), (1, "X"))])

exp_values_tuples = {term: hist.get_expectation_value(term) for term, hist in temp_hist_tuples.items()}
exp_values_strings = {pauli_of_to_string(k, 2): v for k, v in exp_values_tuples.items()}

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

    def test_compute_rdms_from_missing_eigenvalues(self):
        """Compute RDMs from an eigenvalue dictionary should raise and error
        if there is a missing eigenvalue.
        """
        exp_with_missing_values = exp_values_strings.copy()
        exp_with_missing_values.pop("ZZ")
        with self.assertRaises(RuntimeError):
            compute_rdms(ferm_op, "scbk", True, exp_vals=exp_with_missing_values)

    def test_compute_rdms_from_raw_data_strings(self):
        """Compute RDMs from a frequency dictionary (key = strings)."""

        _, _, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, exp_data=exp_data_strings)

        assert_allclose(rdm1ssr, rdm1ss, atol=1e-3)
        assert_allclose(rdm2ssr, rdm2ss, atol=1e-3)

    def test_compute_rdms_from_raw_data_tuples(self):
        """Compute RDMs from a frequency dictionary (key = tuples)."""

        _, _, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, exp_data=exp_data_tuples)

        assert_allclose(rdm1ssr, rdm1ss, atol=1e-3)
        assert_allclose(rdm2ssr, rdm2ss, atol=1e-3)

    def test_compute_rdms_from_eigenvalues_strings(self):
        """Compute RDMs from an eigenvalue dictionary (key = strings)."""

        _, _, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, exp_vals=exp_values_strings)

        assert_allclose(rdm1ssr, rdm1ss, atol=1e-3)
        assert_allclose(rdm2ssr, rdm2ss, atol=1e-3)

    def test_compute_rdms_from_eigenvalues_tuples(self):
        """Compute RDMs from an eigenvalue dictionary (key = tuples)."""

        _, _, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, exp_vals=exp_values_tuples)

        assert_allclose(rdm1ssr, rdm1ss, atol=1e-3)
        assert_allclose(rdm2ssr, rdm2ss, atol=1e-3)

    def test_compute_rdms_from_classical_shadow(self):
        """Compute RDMs from classical shadow"""
        # Construct ClassicalShadow
        bitstrings = []
        unitaries = []

        for b, hist in exp_data_strings.items():
            for s, f in hist.items():
                factor = round(f * 10000)
                bitstrings.extend([s] * factor)
                unitaries.extend([b] * factor)

        cs_data = RandomizedClassicalShadow(unitaries=unitaries, bitstrings=bitstrings)

        _, _, rdm1ss, rdm2ss = compute_rdms(ferm_op, "scbk", True, shadow=cs_data, k=5)

        # Have to adjust tolerance to account for classical shadow rounding to 10000 shots
        assert_allclose(rdm1ssr, rdm1ss, atol=0.05)
        assert_allclose(rdm2ssr, rdm2ss, atol=0.05)

    def test_pad_restricted_rdms_with_frozen_orbitals(self):
        """Test padding of RDMs with frozen orbitals indices (restricted)."""

        mol = mol_H2O_sto3g.freeze_mos([0, 3, 4, 5], inplace=False)

        onerdm_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_sto3g_onerdm_frozen0345.data")
        twordm_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_sto3g_twordm_frozen0345.data")

        test_onerdm, test_twordm = pad_rdms_with_frozen_orbitals_restricted(mol, onerdm_to_pad, twordm_to_pad.reshape((3,)*4))

        padded_onerdm = np.loadtxt(pwd_this_test + "/data/H2O_sto3g_padded_onerdm_frozen0345.data")
        padded_twordm = np.loadtxt(pwd_this_test + "/data/H2O_sto3g_padded_twordm_frozen0345.data")

        np.testing.assert_array_almost_equal(padded_onerdm, test_onerdm, decimal=3)
        np.testing.assert_array_almost_equal(padded_twordm.reshape((7,)*4), test_twordm, decimal=3)

    def test_pad_unrestricted_rdms_with_frozen_orbitals(self):
        """Test padding of RDMs with frozen orbitals indices (unrestricted)."""

        mol = SecondQuantizedMolecule(mol_H2O_sto3g.xyz, uhf=True, frozen_orbitals=[(0, 3, 4, 5), (0, 3, 4, 5)])

        onerdm_a_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_onerdm_frozen0345_alpha.data")
        onerdm_b_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_onerdm_frozen0345_beta.data")
        onerdm_to_pad = (onerdm_a_to_pad, onerdm_b_to_pad)

        twordm_aa_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_twordm_frozen0345_alphaalpha.data")
        twordm_ab_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_twordm_frozen0345_alphabeta.data")
        twordm_bb_to_pad = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_twordm_frozen0345_betabeta.data")
        twordm_to_pad = (twordm_aa_to_pad.reshape((3,)*4), twordm_ab_to_pad.reshape((3,)*4), twordm_bb_to_pad.reshape((3,)*4))

        test_onerdm, test_twordm = pad_rdms_with_frozen_orbitals_unrestricted(mol, onerdm_to_pad, twordm_to_pad)

        padded_onerdm_a = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_padded_onerdm_frozen0345_alpha.data")
        padded_onerdm_b = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_padded_onerdm_frozen0345_beta.data")
        padded_onerdm = (padded_onerdm_a, padded_onerdm_b)

        padded_twordm_aa = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_padded_twordm_frozen0345_alphaalpha.data")
        padded_twordm_ab = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_padded_twordm_frozen0345_alphabeta.data")
        padded_twordm_bb = np.loadtxt(pwd_this_test + "/data/H2O_UHF_sto3g_padded_twordm_frozen0345_betabeta.data")
        padded_twordm = (padded_twordm_aa.reshape((7,)*4), padded_twordm_ab.reshape((7,)*4), padded_twordm_bb.reshape((7,)*4))

        np.testing.assert_array_almost_equal(padded_onerdm, test_onerdm, decimal=3)
        np.testing.assert_array_almost_equal(padded_twordm, test_twordm, decimal=3)


if __name__ == "__main__":
    unittest.main()
