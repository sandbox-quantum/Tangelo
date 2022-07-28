"""
    Python wrappers around IonQ REST API, to facilitate job submission, result retrieval and post-processing
    Using IonQ services requires an API key.
    Users are expected to set the environment variable IONQ_APIKEY with the value of this token.
    IonQ documentation for the API: https://docs.ionq.co
    IonQ cloud dashboard: https://cloud.ionq.com/
"""

import os
import json
import time
import pprint
import pandas as pd
import requests as rq

from tangelo.linq.qpu_connection.qpu_connection import QpuConnection


class IonQConnection(QpuConnection):
    """ Wrapper about the IonQ REST API, to facilitate job submission and automated post-processing of results """

    def __init__(self):
        self.endpoint = "https://api.ionq.co" + "/v0.2"  # Update endpoint or version number here if needed
        self.api_key = None
        self._login()

    @property
    def header(self):
        """ Produce the header for REST requests """
        return {"Content-Type": "application/json", "Authorization": self.api_key}

    def _login(self):
        """ Retrieve the API key to be used for request with IonQ's REST API.
            Assumes users have the environment variable IONQ_APIKEY set to the correct value.
        """
        api_key = os.getenv("IONQ_APIKEY", None)
        if not api_key:
            raise RuntimeError(f"Please set these environment variables: IONQ_APIKEY")
        self.api_key = f'apiKey {api_key}'

        # Verify API key by submitting a trivial request
        try:
            self.job_get_history()
        except RuntimeError as err:
            raise RuntimeError(f"{err}")

    def _catch_request_error(self, return_dict):
        """ Use the dictionary returned from a REST request to check for errors at runtime, and catch them """
        if "error" in return_dict:
            pprint.pprint(return_dict)
            raise RuntimeError(f"Error returned by IonQ API :\n{return_dict['error']}")

    def _get_job_dataframe(self, job_history):
        """ Display main job info as pandas dataframe. Takes REST request answer as input """

        jl = job_history['jobs']
        jl_info = [(j['id'], j['status'], j['target']) for j in jl]
        jobs_df = pd.DataFrame(jl_info, columns=['id', 'status', 'target'])
        return jobs_df

    def job_submit(self, target_backend, ionq_circuit, n_shots, job_name, **job_specs):
        """ Submit job, return job ID.

        Args:
            target_backend (str): name of target device. See IonQ documentation for possible values.
                Current acceptable values are 'simulator' and 'qpu'
            ionq_circuit (str): Circuit in a compatible format (json format recommended)
            n_shots (int): number of shots (ignored if target_backend is set to `simulator`
            job_name (str): name to make the job more identifiable
            **job_specs: extra arguments such as `lang` in the code below; `metadata` is not currently supported.

        Returns:
            job_id (str): string representing the job id
        """

        payload = {"target": target_backend,
                   "name": job_name,
                   "shots": n_shots,
                   "body": ionq_circuit,
                   "lang": job_specs.get('lang', 'json')
                   }

        job_request = rq.post(self.endpoint + '/jobs', headers=self.header, data=json.dumps(payload))
        return_dict = json.loads(job_request.text)

        self._catch_request_error(return_dict)
        print(f"Job submission \tID :: {return_dict['id']} \t status :: {return_dict['status']}")
        return return_dict['id']

    def job_get_history(self):
        """ Returns information about the job corresponding to the input job id

        Args:
            job_id (str): alphanumeric character string representing the job id

        Returns:
            job_status (dict): status response from the REST API
        """

        job_history = rq.get(self.endpoint + '/jobs', headers=self.header)
        return_dict = json.loads(job_history.text)

        self._catch_request_error(return_dict)
        return self._get_job_dataframe(return_dict)

    def job_status(self, job_id):
        """ Returns information about the job corresponding to the input job id

        Args:
            job_id (str): string representing the job id

        Returns:
            job_status (dict): status response from the REST API
        """

        job_status = rq.get(self.endpoint + "/jobs/" + job_id, headers=self.header)
        job_status = json.loads(job_status.text)
        self._catch_request_error(job_status)
        print(f"Job info \tID:: {job_id} \t status :: {job_status['status']} {job_status.get('error', '')}")
        return job_status

    def job_results(self, job_id):
        """ Blocking call querying the REST API at a given frequency, until job results are available.

        Args:
            job_id (str): string representing the job id

        Returns:
            job_status (dict): status response from the REST API
        """

        while True:
            job_status = self.job_status(job_id)
            if job_status['status'] == 'completed' and 'data' in job_status:
                return job_status['data']
            elif job_status['status'] in {'ready', 'running', 'submitted'}:
                time.sleep(5)
            else:
                raise RuntimeError(f'Unexpected job status :: \n {job_status}')

    def job_cancel(self, job_id):
        """ Cancel / delete a job from IonQ servers.

        Args:
            job_id (str): string representing the job id

        Returns:
            job_status (dict): status response from the REST API
        """
        job_cancel = rq.delete(self.endpoint+"/jobs/"+job_id, headers=self.header)
        job_cancel = json.loads(job_cancel.text)
        self._catch_request_error(job_cancel)
        print(f"Job cancel \tID :: {job_id} \t status :: {job_cancel['status']} {job_cancel.get('error', '')}")
        return job_cancel
