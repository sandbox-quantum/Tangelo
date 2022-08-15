"""
    A test class to check that features related to IBM API for quantum experiments are behaving as expected.
    Tests requiring actual interactions with the services are not run automatically.
"""

import unittest
import os
import pprint

from qiskit_ibm_runtime import QiskitRuntimeService
from tangelo.linq import Gate, Circuit
from tangelo.linq.qpu_connection import IBMConnection

circ1 = Circuit([Gate("H", 0), Gate("X", 1)])
res_simulator_circ1 = {'01': 0.5, '11': 0.5}


def assert_freq_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two frequency dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Frequency {k}, difference above tolerance {atol}: {d1[k]} != {d2[k]}")

token = 'c379065b36d454d7bd56a3c804d17a360bda413fb9d448ad5f16a57d103056c24af45efbfe1053e9b858b6a6373ff0733b87fc2d12d7137ad09c6ed6736355a4'
#@unittest.skip("We do not want to store login information for automated testing")
class TestIBMConnection(unittest.TestCase):

    def test_init(self):
        """ Instantiate"""
        QiskitRuntimeService(channel="ibm_quantum", token=token)
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

    def test_submit_job_simulator(self):
        """ Submit a valid job to a API validation backend (simulator) and retrieve results """

        connection = IBMConnection()

        job_id = connection.job_submit('ibmq_qasm_simulator', circ1, 10000)
        job_results = connection.job_results(job_id)
        pprint.pprint(job_results)

        assert_freq_dict_almost_equal(job_results, res_simulator_circ1, 1e-2)

    # def test_delete_job(self):
    #     """ Submit a job and then cancel/delete it, regardless of its status. Check job history before and after. """
    #
    #     ionq_api = IBMConnection()
    #
    #     job_id = ionq_api.job_submit('simulator', circ1, 1000, 'test_simulator_cancel')
    #     job_history_df_before = ionq_api.job_get_history()
    #     assert(job_id in job_history_df_before.id.values)
    #     print(job_history_df_before)
    #     ionq_api.job_cancel(job_id)
    #     job_history_df_after = ionq_api.job_get_history()
    #     assert(job_id not in job_history_df_after.id.values)
    #
    # def test_get_backend_info(self):
    #     """ Retrieve backend info """
    #     ionq_api = IBMConnection()
    #
    #     res = ionq_api.get_backend_info()
    #     pprint.pprint(res)
    #
    # def test_get_characterization(self):
    #     """ Get device characterization through name or charac url """
    #     ionq_api = IBMConnection()
    #     backend = 'qpu.s11'  # Pick something that has a charac url for this test to be useful
    #
    #     res = ionq_api.get_backend_info()
    #     pprint.pprint(res)
    #
    #     # Retrieve charac info from backend name
    #     d1 = ionq_api.get_characterization(backend_name=backend)
    #
    #     # Retrieve charac info from charac url
    #     charac_url = res[res['backend'] == backend]['characterization_url'].iat[0]
    #     d2 = ionq_api.get_characterization(charac_url=charac_url)


if __name__ == "__main__":
    unittest.main()
