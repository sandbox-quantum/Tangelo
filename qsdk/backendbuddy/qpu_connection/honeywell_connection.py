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

"""Python wrappers around Honeywell REST API, to facilitate job submission,
result retrieval and post-processing.

Using Honeywell services require logins to their portal:
https://um.qapi.honeywell.com/index.html
Users are expected to set the environment variables HONEYWELL_EMAIL,
HONEYWELL_PASSWORD with their credentials.

The portal above provides access to a dashboard, which is better suited for job
monitoring experiments.
"""

import os
import re
import json
import time
import pprint
import requests as rq
from collections import Counter

from qsdk.backendbuddy.qpu_connection.qpu_connection import QpuConnection


class HoneywellConnection(QpuConnection):
    """Wrapper about the Honeywell REST API, to facilitate job submission and
    automated post-processing of results.
    """

    def __init__(self):
        self.endpoint = "https://qapi.honeywell.com" + "/v1/"   # Update endpoint or version number here if needed
        self.id_token, self.refresh_token = None, None
        self._login()

    @property
    def header(self):
        """ Produce the header for REST requests """
        return {"Content-Type": "application/json", "Authorization": self.id_token}

    def _login(self):
        """Use Honeywell logins (email, password) to retrieve the id token to be
        used for request with their REST API. Assumes users have set
        HONEYWELL_EMAIL and HONEYWELL_PASSWORD in their environment variables.
        """

        honeywell_email, honeywell_password = os.getenv("HONEYWELL_EMAIL", None), os.getenv("HONEYWELL_PASSWORD", None)
        if not (honeywell_email and honeywell_password):
            raise RuntimeError(f"Please set these environment variables: HONEYWELL_EMAIL, HONEYWELL_PASSWORD")

        login_body = {"email": honeywell_email, "password": honeywell_password}
        login_response = rq.post(self.endpoint+"/login", headers=self.header, data=json.dumps(login_body))
        login_response = json.loads(login_response.text)

        self._catch_request_error(login_response)
        self.id_token, self.refresh_token = login_response["id-token"], login_response["refresh-token"]

    def _catch_request_error(self, return_dict):
        """Use the dictionary returned from a REST request to check for errors
        at runtime, and catch them.
        """
        if "error" in return_dict:
            pprint.pprint(return_dict)
            raise RuntimeError(f"Error returned by Honeywell API :\n{return_dict['error']}")

    def get_devices(self):
        """Return dictionary of available devices to the user, as well as some
        useful information about them.
        """

        device_list_response = rq.get(self.endpoint + "/machine?config=true", headers=self.header)
        available_devices = json.loads(device_list_response.text)
        self._catch_request_error(available_devices)

        # Append useful information regarding said devices (number of qubits, current status...)
        device_info = dict()
        for device in available_devices:
            device_name = device.pop("name") if isinstance(device, dict) else device
            device_state = rq.get(self.endpoint + "/machine/" + device_name, headers=self.header)
            device_state = json.loads(device_state.text)
            self._catch_request_error(device_state)
            device_info[device_name] = {**device, **device_state}
        return device_info

    def job_submit(self, machine_name, qasm_circuit, n_shots, job_name, **job_specs):
        """Submit job, return job ID.

        Args:
            machine_name (str): name of the target device.
            qasm_circuit (str): openqasm 2.0 string representing the quantum
                circuit.
            n_shots (int): number of shots.
            job_name (str): name to make the job more identifiable.
            **job_specs: extra arguments such as `max_cost` or `options` in the
                code below.

        Returns:
            str: alphanumeric character string representing the job id.
        """

        # Honeywell does not support openqasm comments: remove them before submission
        qasm_circuit = re.sub(r'//(.*)\n', '', qasm_circuit)

        body = {"machine": machine_name,
                "name": job_name,
                "count": n_shots,
                "program": qasm_circuit,
                "language": "OPENQASM 2.0",
                "notify": True,
                "max-cost": job_specs.get("max_cost", 100),
                "options": job_specs.get("options", ["no-opt"])
                }

        job_request = rq.post(self.endpoint + "/job/", headers=self.header, data=json.dumps(body))
        job_request = json.loads(job_request.text)

        self._catch_request_error(job_request)
        print(f"Job ID :: {job_request['job']} \t status :: {job_request['status']}")
        return job_request['job']

    def job_get_info(self, job_id):
        """Returns information about the job corresponding to the input job id.

        Args:
            job_id (str): alphanumeric character string representing the job id.

        Returns:
            dict: status response from the Honeywell REST API.
        """

        option = "?websocket=true"
        job_status = rq.get(self.endpoint + "/job/" + job_id + option, headers=self.header)
        job_status = json.loads(job_status.text)

        self._catch_request_error(job_status)
        print(f"Job {job_id} \t status :: {job_status['status']} {job_status.get('error', '')}")
        return job_status

    def job_get_results(self, job_id):
        """Blocking call querying the REST API at a given frequency, until job
        results are available.

        Args:
            job_id (str): alphanumeric character string representing the job id.

        Returns:
            dict: status response from the Honeywell REST API.
        """

        # The only way to retrieve the results, see if a job submission was incorrect, etc, is to look at the job info
        while True:
            job_status = self.job_get_info(job_id)
            if job_status['status'] == 'completed' and 'results' in job_status:
                return job_status['results']
                # TODO: if we know the qubit order, we can return result in standard backendbuddy format later
                #  return job_status['results'] if raw else dict(Counter(job_status['results']['c']))
                #  binary string keys of dictionary may need to be reversed.
            elif job_status['status'] in {'queued', 'running'}:
                time.sleep(10)
            else:
                raise RuntimeError(f'Unexpected job status :: \n {job_status}')

    def job_cancel(self, job_id):
        """Blocking call querying the REST API at a given frequency, until job
        results are available.

        Args:
            str: alphanumeric character string representing the job id.
        """
        job_cancel = rq.post(self.endpoint+"/job/"+job_id+"/cancel", headers=self.header, data={})
        job_cancel = json.loads(job_cancel.text)

        # The operation may fail if it is too late for the job to be canceled
        # In that case, the error message is printed and the error is raised, but we don't follow up on that
        try:
            self._catch_request_error(job_cancel)
        except RuntimeError:
            pass

        time.sleep(3)
        job_status = self.job_get_info(job_id)
        return job_status
