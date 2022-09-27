"""
    Wrappers around Qiskit runtime API, to manage quantum experiments run with IBM Cloud or IBM quantum
    from Tangelo.
"""

import os

from tangelo.linq.translator import translate_qiskit
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection

try:
    from qiskit.providers.jobstatus import JobStatus
    from qiskit_ibm_runtime import QiskitRuntimeService
    is_qiskit_installed = True
except ModuleNotFoundError:
    is_qiskit_installed = False


class IBMConnection(QpuConnection):
    """ Wrapper around IBM Qiskit runtime API to facilitate job submission from Tangelo """

    def __init__(self):

        if not is_qiskit_installed:
            raise ModuleNotFoundError("Both qiskit and qiskit_ibm_runtime need to be installed.")

        self.api_key = None
        self.service = self._login()
        self.jobs = dict()
        self.jobs_results = dict()

    def _login(self):
        """ Attempt to connect to the service. Fails if environment variable IBM_TOKEN
            has not been set to a correct value.
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

    def get_backend_info(self):
        """ Return configuration information for each device found on the service """
        return [b.configuration() for b in self.service.backends()]

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

        # Store job object, return job ID.
        self.jobs[job.job_id] = job
        return job.job_id

    def job_status(self, job_id):
        """ Return information about the job corresponding to the input job ID

        Args:
            job_id (str): string representing the job id

        Returns:
            enum value: status response from the native API
        """
        return self.jobs[job_id].status()

    def job_results(self, job_id):
        """ Blocking call requesting job results.

        Args:
            job_id (str): string representing the job id

        Returns:
            dict: histogram of measurements
        """

        # Retrieve job object, check job has not been cancelled, retrieve results if not
        job = self.jobs[job_id]
        result = job.result()

        if job.status() == JobStatus.CANCELLED:
            print(f"Job {job_id} was cancelled and no results can be retrieved.")
            return None

        self.jobs_results[job_id] = job._results

        # Return histogram for user in standard Tangelo format
        freqs = result['quasi_dists'][0]
        freqs = {bitstring[::-1]: freq for bitstring, freq in freqs.items()}
        return freqs

    def job_cancel(self, job_id):
        """ Attempt to cancel an existing job. May fail depending on job status (e.g too late)

        Args:
            job_id (str): string representing the job id

        Returns:
            bool: whether the job was successfully cancelled.
        """
        job = self.jobs[job_id]
        is_cancelled = True

        try:
            job.cancel()
        except Exception as err:
            is_cancelled = False

        message = "successful" if is_cancelled else "failed"
        print(f"Job {job_id} :: cancellation {message}.")

        return is_cancelled
