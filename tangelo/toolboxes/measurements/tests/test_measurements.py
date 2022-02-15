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

import unittest
import os
from openfermion import load_operator

from tangelo.helpers.utils import default_simulator
from tangelo.linq import translator, Simulator, Circuit
from tangelo.linq.helpers import measurement_basis_gates
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.measurements import get_measurement_estimate

path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'

op1 = 1. * QubitOperator('X1 Y0')
op2 = 0.01 * QubitOperator('Z0 X1')


def assert_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Dictionary entries beyond tolerance {atol}: \n{d1} \n{d2}")
    return True


class MeasurementsTest(unittest.TestCase):

    def test_measurement_uniform(self):
        """ Test uniform measurement estimation method on simple cases """

        # Integer log10 coefficients
        mes1 = get_measurement_estimate(op1)
        mes2 = get_measurement_estimate(op2)
        mes3 = get_measurement_estimate(op2 + 0.1*op1)

        # Float log10 coefficients
        mes4 = get_measurement_estimate(0.6*op1)

        # Coefficient smaller than desired accuracy -> 0
        mes5 = get_measurement_estimate(0.0001*op1)

        # Borderline case: Instead of 1 measurement, take 100.
        mes6 = get_measurement_estimate(0.001*op1)

        # Adjust digits
        mes7 = get_measurement_estimate(op1, digits=2)

        assert_dict_almost_equal(mes1, {((0, 'Y'), (1, 'X')): 100000000}, 1)
        assert_dict_almost_equal(mes2, {((0, 'Z'), (1, 'X')): 10000}, 1)
        assert_dict_almost_equal(mes3, {((0, 'Z'), (1, 'X')): 10000, ((0, 'Y'), (1, 'X')): 1000000}, 1)
        assert_dict_almost_equal(mes4, {((0, 'Y'), (1, 'X')): 36000000}, 1)
        assert_dict_almost_equal(mes5, {((0, 'Y'), (1, 'X')): 0}, 1)
        assert_dict_almost_equal(mes6, {((0, 'Y'), (1, 'X')): 100}, 1)
        assert_dict_almost_equal(mes7, {((0, 'Y'), (1, 'X')): 1000000}, 1)

    def test_measurement_uniform_H2(self):
        """ Test on UCCSD H2 usecase that uniform measurement estimation method guarantees on average the level
        of accuracy expected by the user """

        # Load state preparation circuit
        with open(f"{path_data}/H2_UCCSD.qasm", "r") as f:
            openqasm_circ = f.read()
        abs_circ = translator._translate_openqasm2abs(openqasm_circ)

        # Load qubit Hamiltonian
        qb_ham = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

        # Get exact expectation value using a simulator
        sim_exact = Simulator()
        freqs_exact, _ = sim_exact.simulate(abs_circ)
        exp_val_exact = sim_exact.get_expectation_value(qb_ham, abs_circ)

        # Get measurement estimate using "uniform" method
        mes_dict = get_measurement_estimate(qb_ham, digits=2)

        # Get samples and compute expectation value from measurements, using exact state preparation. Compute error.
        diffs = []
        n_repeat = 50
        for i in range(n_repeat):
            exp_val = 0.
            sim = Simulator(target=default_simulator, n_shots=1)
            for m_basis, coef in qb_ham.terms.items():
                if m_basis:
                    term_circ = abs_circ + Circuit(measurement_basis_gates(m_basis))
                    sim.n_shots = mes_dict[m_basis]
                    freqs, _ = sim.simulate(term_circ)
                    exp_val += sim.get_expectation_value_from_frequencies_oneterm(m_basis, freqs) * coef
                else:
                    exp_val += coef
            diffs.append(abs(exp_val - exp_val_exact))

        # Check that on average, we deliver the expected accuracy
        average_diff = sum(diffs)/n_repeat
        assert(average_diff <= 1e-2)


if __name__ == "__main__":
    unittest.main()
