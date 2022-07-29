"""
    A test class to check that features related to the Honeywell API are behaving as expected.
    Tests requiring actual interactions with the services have been skipped.
"""

import unittest
import os
import pprint

from tangelo.linq import Gate, Circuit, translator
from tangelo.linq.qpu_connection import IonQConnection

circ1 = Circuit([Gate("H", 0), Gate("CNOT", target=1, control=0)])
json_circ1 = translator.translate_json_ionq(circ1)
res_simulator_circ1 = {'histogram': {'0': 0.5, '3': 0.5}}


def assert_freq_dict_almost_equal(d1, d2, atol):
    """ Utility function to check whether two frequency dictionaries are almost equal, for arbitrary tolerance """
    if d1.keys() != d2.keys():
        raise AssertionError("Dictionary keys differ. Frequency dictionaries are not almost equal.\n"
                             f"d1 keys: {d1.keys()} \nd2 keys: {d2.keys()}")
    else:
        for k in d1.keys():
            if abs(d1[k] - d2[k]) > atol:
                raise AssertionError(f"Frequency {k}, difference above tolerance {atol}: {d1[k]} != {d2[k]}")


@unittest.skip("We do not want to store login information for automated testing")
class TestIonQConnection(unittest.TestCase):

    def test_init(self):
        """ If user has set environment variables IONQ_APIKEY to the correct value, this should succeed.
        Implicitly makes a call to IonQConnection.job_get_history in order to validate the apiKey. """
        IonQConnection()

    def test_init_fail(self):
        """ If user has not set environment variables IONQ_APIKEY to the correct value, this should
        return a RuntimeError. """
        tmp = os.getenv("IONQ_APIKEY", '')
        os.environ['IONQ_APIKEY'] = 'invalid_apikey'
        self.assertRaises(RuntimeError, IonQConnection)
        os.environ['IONQ_APIKEY'] = ''
        self.assertRaises(RuntimeError, IonQConnection)
        os.environ['IONQ_APIKEY'] = tmp

    def test_submit_job_simulator(self):
        """ Submit a valid job to a API validation backend (simulator) and retrieve results """

        ionq_api = IonQConnection()

        job_id = ionq_api.job_submit('simulator', json_circ1, 100, 'test_simulator_json_job')
        job_results = ionq_api.job_results(job_id)
        pprint.pprint(job_results)

        assert_freq_dict_almost_equal(job_results['histogram'], res_simulator_circ1['histogram'], 1e-7)

    def test_delete_job(self):
        """ Submit a job and then cancel/delete it, regardless of its status. Check job history before and after. """

        ionq_api = IonQConnection()

        job_id = ionq_api.job_submit('simulator', json_circ1, 1000, 'test_simulator_json_job')
        job_history_df_before = ionq_api.job_get_history()
        assert(job_id in job_history_df_before.id.values)
        print(job_history_df_before)
        ionq_api.job_cancel(job_id)
        job_history_df_after = ionq_api.job_get_history()
        assert(job_id not in job_history_df_after.id.values)

    def test_get_backend_info(self):
        """ Retrieve backend info """
        ionq_api = IonQConnection()

        res = ionq_api.get_backend_info()
        pprint.pprint(res)

    def test_get_characterization(self):
        """ Get device characterization through name or charac url """
        ionq_api = IonQConnection()
        backend = 'qpu.s11' # Pick something that has a charac url for this test to be useful

        res = ionq_api.get_backend_info()
        pprint.pprint(res)

        # Retrieve charac info from backend name
        d1 = ionq_api.get_characterization(backend_name=backend)

        # retrieve charac info from charac url
        charac_url = res[res['backend'] == backend]['characterization_url'].iat[0]
        d2 = ionq_api.get_characterization(charac_url=charac_url)


if __name__ == "__main__":
    unittest.main()