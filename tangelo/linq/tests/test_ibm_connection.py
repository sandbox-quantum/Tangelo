"""
    A test class to check that features related to IBM API for quantum experiments are behaving as expected.
    Tests requiring actual interactions with the services are not run automatically.
"""

import unittest
import os
import time

from tangelo.linq import Gate, Circuit
from tangelo.linq.qpu_connection import IBMConnection
from tangelo.toolboxes.operators import QubitOperator

circ = Circuit([Gate("H", 0), Gate("X", 1)])
op = 1.0 * QubitOperator('Y0') - 2.0 * QubitOperator('Z0 X1')

res_sampler = {'01': 0.5, '11': 0.5}
res_estimator = {}

def assert_freq_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two frequency dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Frequency {k}, difference above tolerance {atol}: {d1[k]} != {d2[k]}")


os.environ["IBM_TOKEN"] = 'c379065b36d454d7bd56a3c804d17a360bda413fb9d448ad5f16a57d103056c24af45efbfe1053e9b858b6a6373ff0733b87fc2d12d7137ad09c6ed6736355a4'
#@unittest.skip("We do not want to store login information for automated testing. Manual testing only.")
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
        os.environ['IBM_TOKEN'] = ''
        self.assertRaises(RuntimeError, IBMConnection)
        os.environ['IBM_TOKEN'] = tmp

    def test_submit_job_sampler(self):
        """ Submit a sampler job to a valid backend, query status and retrieve results """

        connection = IBMConnection()

        #args = {'ibmq_qasm_simulator', circ, n_shots=10**5}

        job_id = connection.job_submit('sampler', 'ibmq_qasm_simulator', circ, n_shots=10**5)
        print(connection.job_status(job_id))

        job_results = connection.job_results(job_id)
        print(connection.job_status(job_id))

        assert_freq_dict_almost_equal(job_results, res_sampler, 1e-2)

    def test_submit_job_estimator(self):
        """ Submit an estimator job to a valid backend, query status and retrieve results """

        connection = IBMConnection()

        job_id = connection.job_submit('estimator', 'ibmq_qasm_simulator', circ, operator=op, n_shots=10**5)
        print(connection.job_status(job_id))

        job_results = connection.job_results(job_id)
        print(connection.job_status(job_id))

        assert_freq_dict_almost_equal(job_results, res_estimator, 1e-2)

    def test_cancel_job(self):
        """ Submit a job to a valid backend, attempt to cancel """

        connection = IBMConnection()

        for sleep_time in [0., 20.]:
            job_id = connection.job_submit("sampler", 'ibmq_qasm_simulator', circ1*10, 10000)
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
