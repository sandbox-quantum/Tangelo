"""
    A test class to check that features related to Braket SDK for quantum experiments are behaving as expected.
    Tests requiring actual interactions with the services are not run automatically.
"""

import os
import time
import unittest

from tangelo.linq import Gate, Circuit
from tangelo.linq.qpu_connection.braket_connection import BraketConnection
from tangelo.helpers.utils import assert_freq_dict_almost_equal

# Circuit and qubit operator for test, with reference values
circ = Circuit([Gate("H", 0), Gate("X", 1)])
ref = {'01': 0.5, '11': 0.5}

# Set sv1 device arn for tests
sv1_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

# Set up S3 bucket for tests
s3_bucket = "amazon-braket-gc-quantum-dev"  # bucket name
folder = "tangelo_test"  # destination folder

# Set up credentials [clear before pushing to public repo]
os.environ["AWS_REGION"] = ""
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_SESSION_TOKEN"] = ""


@unittest.skip("Manual testing only to avoid uploading login info.")
class TestBraketConnection(unittest.TestCase):

    def test_init(self):
        """ Attempt to instantiate connection """
        BraketConnection()

    def test_submit_job(self):
        """ Submit a simple job to a simulator backend, query status and retrieve results """

        conn = BraketConnection(s3_bucket=s3_bucket, folder=folder)

        job_id = conn.job_submit(sv1_arn, 100000, circ)
        print(conn.job_status(job_id))

        freqs = conn.job_results(job_id)
        print(conn.job_status(job_id))

        assert_freq_dict_almost_equal(freqs, ref, 1e-2)

    def test_submit_batch_job(self):
        """ Submit a batch job (several circuits) to a simulator backend. """

        conn = BraketConnection(s3_bucket=s3_bucket, folder=folder)

        # Retrieve list of job_ids, check they are logged in "jobs" attribute
        job_ids = conn.job_submit(sv1_arn, 100, [circ]*2)
        print(conn.jobs)

        # Ensure individual jobs can be accessed as usual
        for job_id in job_ids:
            conn.job_results(job_id)
            conn.job_status(job_id)

    def test_cancel_job(self):
        """ Submit a job to a valid backend, attempt to cancel """

        conn = BraketConnection(s3_bucket=s3_bucket, folder=folder)

        for sleep_time in [0., 20.]:
            job_id = conn.job_submit(sv1_arn, 100000, circ)
            time.sleep(sleep_time)
            conn.job_cancel(job_id)
            print(conn.job_status(job_id))
            print(conn.job_results(job_id))

    def test_get_backend_info(self):
        """ Return backend info of supported providers """

        conn = BraketConnection()
        print(conn.get_backend_info())


if __name__ == "__main__":
    unittest.main()
