"""
    Wrappers around Qiskit runtime API, to manage quantum experiments run with IBM Cloud or IBM quantum
    from Tangelo.
"""

import os

from tangelo.linq.translator import translate_operator, translate_circuit
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection

try:
    from qiskit.providers.jobstatus import JobStatus
    from qiskit.primitives import SamplerResult, EstimatorResult
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
    is_qiskit_installed = True
except ModuleNotFoundError:
    is_qiskit_installed = False


class IBMConnection(QpuConnection):
    """ Wrapper around IBM Qiskit runtime API to facilitate job submission from Tangelo """

    def __init__(self, ibm_quantum_token=None):

        if not is_qiskit_installed:
            raise ModuleNotFoundError("Both qiskit and qiskit_ibm_runtime need to be installed.")

        self.api_key = ibm_quantum_token if ibm_quantum_token else os.getenv("IBM_TOKEN", None)
        self.service = self._login()
        self.jobs = dict()
        self.jobs_results = dict()

    def _login(self):
        """ Attempt to connect to the service. Fails if environment variable IBM_TOKEN
            has not been set to a correct value.
        """
        if not self.api_key:
            raise RuntimeError(f"Please provide IBM_TOKEN (as environment variable or at instantiation of connection.")
        try:
            return QiskitRuntimeService(channel="ibm_quantum", token=self.api_key)
        except Exception as err:
            raise RuntimeError(f"{err}")

    def get_backend_info(self):
        """ Return configuration information for each device found on the service """
        return {b.name: b.configuration() for b in self.service.backends()}

    def job_submit(self, program, backend_name, n_shots, circuits, operators=None, runtime_options=None, instance=None):
        """ Submit job, return job ID.

        Args:
            program (str): name of available qiskit-runtime program (e.g sampler, estimator currently)
            backend_name (str): name of a qiskit backend
            n_shots (int): Number of shots to use on the target backend
            circuits (Circuit | List[Circuit]): Tangelo circuit(s)
            operators (QubitOperator | List[QubitOperator]) : Optional, qubit operators for computing expectation values
            runtime_options (dict): Optional, extra keyword arguments for options supported in qiskit-runtime.
            instance (str): Optional, desired IBM service instance in the "hub/group/project" format. Default is likely to send to "ibm-q/open/main"

        Returns:
            str: string representing the job id
        """

        # Set up options and intermediary Qiskit runtime objects
        backend = self.service.backend(backend_name, instance=instance)
        session = Session(service=self.service, backend=backend)

        if runtime_options is None:
            runtime_options = dict()
        options = Options(optimization_level=runtime_options.get('optimization_level', 1),
                          resilience_level=runtime_options.get('resilience_level', 0))
        options.execution.shots = n_shots

        # Translate circuits in qiskit format, add final measurements
        if not isinstance(circuits, list):
            circuits = [circuits]
        qiskit_cs = list()
        for c in circuits:
            qiskit_c = translate_circuit(c, target="qiskit")
            qiskit_c.remove_final_measurements()
            qiskit_c.measure_all(add_bits=False)
            qiskit_cs.append(qiskit_c)

        # If needed, translate qubit operators in qiskit format
        if operators:
            if not isinstance(operators, list):
                operators = [operators]
            qiskit_ops = [translate_operator(op, source="tangelo", target="qiskit") for op in operators]

        # Execute qiskit-runtime program, retrieve job ID
        if program == 'sampler':
            sampler = Sampler(session=session, options=options)
            job = sampler.run(circuits=qiskit_cs)
        elif program == 'estimator':
            estimator = Estimator(session=session, options=options)
            job = estimator.run(circuits=qiskit_cs, observables=qiskit_ops)
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

        if job.status() == JobStatus.CANCELLED:
            print(f"Job {job_id} was cancelled and no results can be retrieved.")
            return None

        result = job.result()
        self.jobs_results[job_id] = job._results

        # Sampler: return histogram for user in standard Tangelo format
        if isinstance(result, SamplerResult):
            histograms = []
            for j in range(len(result.quasi_dists)):

                hist = result.quasi_dists[j]
                n_qubits = job.inputs['circuits'][j].num_qubits

                freqs = dict()
                for i, freq in hist.items():
                    bs = bin(i).split('b')[-1]
                    state_binstr = "0" * (n_qubits - len(bs)) + bs
                    freqs[state_binstr[::-1]] = freq
                histograms.append(freqs)
            return histograms

        # Estimator: return the array of expectation values
        elif isinstance(result, EstimatorResult):
            return list(result.values)

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
