"""
    A test class to check that features related to IBM API for quantum experiments are behaving as expected.
    Tests requiring actual interactions with the services are not run automatically.
"""

import unittest
import os
import time

import numpy as np

from tangelo.helpers.utils import assert_freq_dict_almost_equal
from tangelo.linq import Gate, Circuit, get_backend
from tangelo.linq.qpu_connection.ibm_connection import IBMConnection
from tangelo.toolboxes.operators import QubitOperator

# Circuit and qubit operator for test
circ = Circuit([Gate("H", 0), Gate("X", 1)])
circ2 = Circuit([Gate("RX", 0, parameter=2.), Gate("RY", 1, parameter=-1.)])
op = 1.0 * QubitOperator('Y0') - 2.0 * QubitOperator('Z0 X1')

# Reference values
ref_sampler = {'01': 0.5, '11': 0.5}
sim = get_backend()
ref_estimator = sim.get_expectation_value(op, circ2)

os.environ['IBM_TOKEN'] = 'INSERT VALID TOKEN HERE, FOR TESTS. REMOVE AFTERWARDS'


@unittest.skip("We do not want to store login information for automated testing. Manual testing only.")
class TestIBMConnection(unittest.TestCase):

    def test_init(self):
        """ Attempt to instantiate connection (requires valid credentials)"""
        IBMConnection()

    def test_init_fail(self):
        """ If user has not set environment variables IONQ_APIKEY to the correct value, this should
        return a RuntimeError. """
        tmp = os.getenv("IBM_TOKEN", '')
        os.environ['IBM_TOKEN'] = 'invalid_apikey'
        self.assertRaises(RuntimeError, IBMConnection)
        self.assertRaises(RuntimeError, IBMConnection, 'invalid')
        os.environ['IBM_TOKEN'] = ''
        self.assertRaises(RuntimeError, IBMConnection)
        os.environ['IBM_TOKEN'] = tmp

    def test_submit_job_sampler(self):
        """ Submit a sampler job to a valid backend, query status and retrieve results """

        connection = IBMConnection()

        options = {'resilience_level': 1}
        job_id = connection.job_submit('sampler', 'ibmq_qasm_simulator', 10**5, circ, runtime_options=options)
        print(connection.job_status(job_id))

        job_results = connection.job_results(job_id)
        print(connection.job_status(job_id))

        assert_freq_dict_almost_equal(job_results, ref_sampler, 1e-2)

    def test_submit_job_estimator(self):
        """ Submit an estimator job to a valid backend, query status and retrieve results """

        conn = IBMConnection()

        options = {'resilience_level': 1}
        job_id = conn.job_submit('estimator', 'ibmq_qasm_simulator', 10**5, circ2, operators=op, runtime_options=options)
        print(conn.job_status(job_id))

        job_results = conn.job_results(job_id)
        self.assertAlmostEqual(job_results[0], ref_estimator, delta=1e-2)

    def test_submit_job_estimator_list(self):
        """ Submit an estimator job to a valid backend, query status and retrieve results """

        conn = IBMConnection()

        job_id = conn.job_submit('estimator', 'ibmq_qasm_simulator', 10**5, [circ2]*2, operators=[op]*2)
        print(conn.job_status(job_id))

        job_results = conn.job_results(job_id)
        np.testing.assert_almost_equal(np.array(job_results), np.array([ref_estimator]*2), decimal=2)

    def test_cancel_job(self):
        """ Submit a job to a valid backend, attempt to cancel """

        connection = IBMConnection()

        for sleep_time in [0., 20.]:
            job_id = connection.job_submit('sampler', 'ibmq_qasm_simulator', 10**2, circ)
            time.sleep(sleep_time)
            print(connection.job_cancel(job_id))
            print(connection.job_status(job_id))

            job_results = connection.job_results(job_id)

    def test_get_backend_info(self):
        """ Return a list of "configuration" objects for all devices found on the service """

        connection = IBMConnection()
        res = connection.get_backend_info()
        print(res)


if __name__ == "__main__":
    unittest.main()
