"""
    Wrappers around Qiskit runtime API, to manage quantum experiments run with IBM Cloud or IBM quantum
    from Tangelo
"""

import os
import json
import requests as rq

from tangelo.linq.translator import translate_qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection

# TODO: get_backends, job_cancel, main


class IBMConnection(QpuConnection):
    """ Wrapper around IBM Qiskit runtime API to facilitate job submission from Tangelo """

    def __init__(self):
        self.api_key = None
        self.service = self._login()
        self.jobs = dict()
        self.jobs_results = dict()

    def _login(self):
        """ Retrieve the API key to be used for request with IonQ's REST API.
            Assumes users have the environment variable IBM_TOKEN set to the correct value.
        """
        api_key = os.getenv("IBM_TOKEN", None).strip('\'')
        if not api_key:
            raise RuntimeError(f"Please set these environment variables: IBM_TOKEN")
        self.api_key = api_key
        # Verify API key by instantiating the connection
        try:
            return QiskitRuntimeService(channel="ibm_quantum", token=api_key)
        except Exception as err:
            raise RuntimeError(f"{err}")

    def job_submit(self, target_backend, circ, n_shots):
        """ Submit job, return job ID.

        Args:
            target_backend (str): name of target device. See Qiskit documentation for available devices
            circ (Circuit): Circuit in Tangelo format
            n_shots (int): number of shots

        Returns:
            str: string representing the job id
        """

        # Convert Tangelo circuit to Qiskit circuit
        ibm_circ = translate_qiskit(circ)

        # Circuit needs final measurements
        ibm_circ.remove_final_measurements()
        ibm_circ.measure_all(add_bits=False)

        options = {"backend_name": target_backend}
        run_options = {"shots": n_shots}

        # Future extra keywords and feature to support, error-mitigation and optimization of circuit
        resilience_settings = {"level": 0}  # Default: no error-mitigation, raw results.

        program_inputs = {"circuits": ibm_circ, "circuit_indices": [0],
                          "run_options": run_options, "resilience_settings": resilience_settings}

        job = self.service.run(program_id="sampler", options=options, inputs=program_inputs)

        # Keep job object
        self.jobs[job.job_id] = job
        return job.job_id

    def job_status(self, job_id):
        """ Returns information about the job corresponding to the input job id

        Args:
            job_id (str): string representing the job id

        Returns:
            enum value: status response from the native API
        """
        return self.jobs[job_id].status()

    def job_results(self, job_id):
        """ Blocking call querying the REST API at a given frequency, until job results are available.

        Args:
            job_id (str): string representing the job id

        Returns:
            result object from the IBM runtime API
        """

        # Retrieve job object, request results and creates a reference to Qiskit runtime raw results
        job = self.jobs[job_id]
        result = job.result()
        self.jobs_results[job_id] = result #job._results

        # Return histogram for user in standard Tangelo format
        freqs = result['quasi_dists'][0]
        freqs = {bitstring[::-1]: freq for bitstring, freq in freqs.items()}
        return freqs
