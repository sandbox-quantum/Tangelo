# Copyright 2021 1QB Information Technologies Inc.
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

"""
    A test class to check that features related to the Honeywell API are behaving as expected.
    Tests requiring actual interactions with the services have been skipped.
"""

import unittest
import os

from qsdk.backendbuddy import Gate, Circuit, translate_openqasm
from qsdk.backendbuddy.qpu_connection import HoneywellConnection


circ1 = Circuit([Gate("H", 0), Gate("CNOT", target=1, control=0)])
circ1_qasm = translate_openqasm(circ1)


class TestHoneywellConnection(unittest.TestCase):

    @unittest.skip("We do not want to store login information for automated testing")
    def test_init(self):
        """ If the user has set environment variables HONEYWELL_EMAIL and HONEYWELL_PASSWORD to the correct login
         information, this test should succeed. """
        HoneywellConnection()

    def test_init_fail(self):
        """ If the user has not set environment variables HONEYWELL_EMAIL and HONEYWELL_PASSWORD to the correct login
         information, this should fail and return an EnvironmentError. """
        tmp = os.getenv("HONEYWELL_EMAIL", 'empty')
        os.environ['HONEYWELL_EMAIL'] = 'dummy_email'
        self.assertRaises(RuntimeError, HoneywellConnection)
        os.environ['HONEYWELL_EMAIL'] = tmp

    @unittest.skip("We do not want to store login information for automated testing")
    def test_submit_job(self):
        """ Submit a valid job to a API validation backend (if available) and retrieve results """

        honeywell_api = HoneywellConnection()
        devices = honeywell_api.get_devices()
        print(f"{devices}")
        validation_devices = [d for d in devices if d.endswith('APIVAL')]
        print(validation_devices)

        if not validation_devices:
            print("No `APIVAL` (validation) device currently available through Honeywell. Ending job submission test.")
            return

        n_shots = 10
        job_id = honeywell_api.job_submit(validation_devices[0], circ1_qasm, n_shots, '1qbit_test_submit_job')
        results = honeywell_api.job_get_results(job_id)
        self.assertEqual(results, {'00': n_shots})  # Validation backend does not perform any simulation

    @unittest.skip("We do not want to store login information for automated testing")
    def test_submit_job_incorrect_circuit(self):
        """ Submit an incorrect job to a API validation backend (if available) and retrieve results """

        circ_qasm = "#$%+" + translate_openqasm(circ1)

        honeywell_api = HoneywellConnection()
        devices = honeywell_api.get_devices()
        print(f"{devices}")
        validation_devices = [d for d in devices if d.endswith('APIVAL')]
        print(validation_devices)

        if not validation_devices:
            print("No `APIVAL` (validation) device currently available through Honeywell. Ending job submission test.")
            return

        n_shots = 10
        job_id = honeywell_api.job_submit(validation_devices[0], circ_qasm, n_shots, '1qbit_test_submit_job')
        self.assertRaises(RuntimeError, honeywell_api.job_get_results, job_id)


if __name__ == "__main__":
    unittest.main()
