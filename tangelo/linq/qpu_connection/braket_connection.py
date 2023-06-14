"""
    Wrappers around Braket API, to manage quantum experiments run with Braket from Tangelo.
"""

from tangelo.linq.translator import translate_operator, translate_circuit
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection

try:
    import braket
    from braket.aws import AwsDevice, AwsQuantumTask
    is_braket_installed = True
except ModuleNotFoundError:
    is_braket_installed = False


class BraketConnection(QpuConnection):
    """ Wrapper around Amazon Braket API to facilitate quantum job management from Tangelo """

    def __init__(self, s3_bucket=None, folder=None):

        if not is_braket_installed:
            raise ModuleNotFoundError("Braket needs to be installed.")

        self.jobs = dict()
        self.jobs_results = dict()
        self.s3_bucket = s3_bucket
        self.folder = folder

    def get_backend_info(self):
        """ Return configuration information for each device found on the service """
        return AwsDevice.get_devices()

    def job_submit(self, backend_arn, n_shots, circuit):
        """ Submit job, return job ID.

        Args:
            backend_name (str): name of a qiskit backend
            n_shots (int): Number of shots to use on the target backend
            circuits (Circuit | List[Circuit]): Tangelo circuit(s)
            str: string representing the job id
        """

        # Set up options and intermediary Qiskit runtime objects
        device = AwsDevice(backend_arn)

        # Translate circuits in braket format
        braket_c = translate_circuit(circuit, "braket")

        # Ensure s3 location for results is set to a valid value
        if not (self.s3_bucket and self.folder):
            raise ValueError(f"{self.__class__.__name__} :: Please set the following attributes: s3_bucket, folder")
        s3_location = (self.s3_bucket, self.folder)

        # Submit task
        my_task = device.run(braket_c, s3_location, shots=1000, poll_timeout_seconds=100, poll_interval_seconds=10)

        # Store job object, return job ID.
        self.jobs[my_task.id] = my_task
        return my_task.id

    def job_status(self, job_id):
        """ Return information about the job corresponding to the input job ID

        Args:
            job_id (str): string representing the job id

        Returns:
            enum value: status response from the native API
        """
        return self.jobs[job_id].state()

    def job_results(self, job_id):
        """ Blocking call requesting job results.

        Args:
            job_id (str): string representing the job id

        Returns:
            dict: histogram of measurements
        """

        # Retrieve job object
        if job_id in self.jobs:
            job = self.jobs[job_id]
        else:
            job = AwsQuantumTask(arn=job_id)

        # Check status of job, raise error if results cannot be retrieved
        if self.job_status(job_id) in {'CANCELLED', 'FAILED'}:
            print(f"Job {job_id} was cancelled or failed, and no results can be retrieved.")
            return None

        # Retrieve results, update job dictionary
        result = job.result()
        self.jobs_results[job_id] = result
        return result

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

    # def _submit_sampler(self, qiskit_c, n_shots, session, options):
    #     """ Submit job using Sampler primitive, return job ID.
    #
    #     Args:
    #         qiskit_c (Qiskit.QuantumCircuit): Circuit in Qiskit format
    #         n_shots (int): Number of shots
    #         session (qiskit_ibm_runtime.Session): Qiskit runtime Session object
    #         options (qiskit_ibm_runtime.Options): Qiskit runtime Options object
    #
    #     Returns:
    #         str: string representing the job id
    #     """
    #
    #     # Set up program inputs
    #     run_options = {"shots": n_shots}
    #     resilience_settings = {"level": options.resilience_level}
    #
    #     program_inputs = {"circuits": qiskit_c, "circuit_indices": [0],
    #                       "run_options": run_options,
    #                       "resilience_settings": resilience_settings}
    #
    #     # Set backend
    #     more_options = {"backend_name": session.backend()}
    #
    #     job = self.service.run(program_id="sampler", options=more_options, inputs=program_inputs)
    #     return job
