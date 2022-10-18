"""
    Wrappers around Qiskit runtime API, to manage quantum experiments run with IBM Cloud or IBM quantum
    from Tangelo.
"""

import os

from tangelo.linq.translator import translate_operator, translate_c_to_qiskit
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection

try:
    from qiskit.providers.jobstatus import JobStatus
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options, \
        SamplerResult, EstimatorResult
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

    def job_submit(self, program, backend, circ, **kwargs):
        """ Submit job, return job ID.

        Args:
            program (str): name of qiskit-runtime program (e.g sampler, estimator....)
            backend (str): name of a qiskit backend
            circ (Circuit): Tangelo circuit
            **kwargs (dict): extra keyword arguments. See body for what is currently supported.

        Returns:
            str: string representing the job id
        """

        n_shots = kwargs.get('n_shots', 10**4)
        op = kwargs.get('operator', None)

        if program == 'sampler':
            job = self._submit_sampler(backend, circ, n_shots)
        elif program == 'estimator':
            job = self._submit_estimator(backend, circ, op, n_shots)
        else:
            raise NotImplementedError("Only Sampler and Estimator programs currently available.")

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

        # Sampler: return histogram for user in standard Tangelo format
        if isinstance(result, SamplerResult):
            hist = result.quasi_dists[0]

            freqs = dict()
            for i, freq in hist.items():
                bs = bin(i).split('b')[-1]
                n_qubits = job.inputs['circuits'].num_qubits
                state_binstr = "0" * (n_qubits - len(bs)) + bs
                freqs[state_binstr[::-1]] = freq
            return freqs

        # Estimator: return the expectation value
        elif isinstance(result, EstimatorResult):
            return result.values[0]

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

    def _submit_sampler(self, backend_name, circuit, n_shots):
        """ Submit job using Sampler primitive, return job ID.

        Args:
            backend_name (str): name of a qiskit backend
            circuit (Circuit): Tangelo circuit
            n_shots (int): Number of shots

        Returns:
            str: string representing the job id
        """

        # Translate circuit in qiskit format, add final measurements
        qiskit_c = translate_c_to_qiskit(circuit)
        qiskit_c.remove_final_measurements()
        qiskit_c.measure_all(add_bits=False)

        options = {"backend_name": backend_name}
        run_options = {"shots": n_shots}

        # Future extra keywords and feature to support, error-mitigation and optimization of circuit
        resilience_settings = {"level": 0}  # Default: no error-mitigation, raw results.

        program_inputs = {"circuits": qiskit_c, "circuit_indices": [0],
                          "run_options": run_options, "resilience_settings": resilience_settings}

        job = self.service.run(program_id="sampler", options=options, inputs=program_inputs)
        return job

    def _submit_estimator(self, backend_name, circuit, operator, n_shots):
        """ Submit job using Estimator primitive, return job ID.

        Args:
            backend_name (str): name of a qiskit backend
            circuit (Circuit): Tangelo circuit
            operator (QubitOperator): Tangelo QubitOperator
            n_shots (int): Number of shots

        Returns:
            str: string representing the job id
        """

        # Translate circuit in qiskit format, add final measurements
        qiskit_c = translate_c_to_qiskit(circuit)
        qiskit_c.remove_final_measurements()
        qiskit_c.measure_all(add_bits=False)

        # Translate qubit operator in qiskit format
        qiskit_op = translate_operator(operator, source="tangelo", target="qiskit")

        # Set up options and intermediary objects
        options = {}
        backend = self.service.backend(backend_name)
        session = Session(service=self.service, backend=backend)
        estimator = Estimator(session=session, options=options)

        # Submit job to estimator
        job = estimator.run(circuits=[qiskit_c], observables=[qiskit_op], shots=n_shots)
        return job
