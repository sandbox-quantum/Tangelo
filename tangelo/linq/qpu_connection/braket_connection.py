"""
    Wrappers around Braket python's API, to manage quantum experiments run with Braket from Tangelo.
    Braket is available at https://github.com/aws/amazon-braket-sdk-python
"""

from enum import Enum

from tangelo.linq.translator import translate_circuit
from tangelo.linq.qpu_connection.qpu_connection import QpuConnection


class SupportedBraketProviders(str, Enum):
    """ List of the providers currently supported. Needs to be occasionally
    updated in the future as new providers arise. """

    # SV1, TN1 and DM1 devices (simulators)
    AMAZON = "Amazon Braket"

    # Hardware providers
    IONQ = "IonQ"
    RIGETTI = "Rigetti"
    OXFORD = "Oxford"


def refresh_available_braket_devices():
    """Function to get the available gate-based devices on Braket.
    Note: OFFLINE and RETIRED devices are filtered out.
    Returns:
        list of braket.aws.AwsDevice: Available gate-based Braket devices.
    """
    from braket.aws import AwsDevice

    return AwsDevice.get_devices(
        provider_names=[provider.value for provider in SupportedBraketProviders],
        statuses=["ONLINE"])


class BraketConnection(QpuConnection):
    """ Wrapper around Amazon Braket API to facilitate quantum job management from Tangelo """

    def __init__(self, s3_bucket=None, folder=None):
        """
        Initialize connection object. Requires Braket to be installed (see above).
        The destination s3_bucket and folder can be specified later, before submitting jobs.

        Args:
            s3_bucket (str): Optional, name of target s3_bucket for saving results
            folder (str): Optional, name of target folder in bucket for results

        Returns:
            str | List[str]: Job id(s)
        """

        try:
            import braket
            self.braket = braket
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Braket needs to be installed.")

        self.jobs = dict()
        self.jobs_results = dict()
        self.s3_bucket = s3_bucket
        self.folder = folder

    def job_submit(self, backend_arn, n_shots, circuits):
        """ Submit job as batch, return job id(s).

        Args:
            backend_arn (str): arn for braket backend
            n_shots (int): Number of shots to use on the target backend
            circuits (Circuit | List[Circuit]): Tangelo circuit(s)

        Returns:
            str | List[str]: Job id(s)
        """

        # Set up device object
        from braket.aws import AwsDevice
        device = AwsDevice(backend_arn)

        # Ensure s3 location for results is set to a valid value
        if not (self.s3_bucket and self.folder):
            raise ValueError(f"{self.__class__.__name__} :: Please set the following attributes: s3_bucket, folder")
        s3_location = (self.s3_bucket, self.folder)

        # Ensure input is a list of circuits
        if not isinstance(circuits, list):
            circuits = [circuits]

        # Translate circuits in braket format, submit as batch
        braket_circuits = [translate_circuit(c, "braket") for c in circuits]
        my_task = device.run_batch(braket_circuits, s3_location, shots=n_shots,
                                   poll_timeout_seconds=300, poll_interval_seconds=60)

        # Store job object(s)
        for t in my_task.tasks:
            self.jobs[t.id] = t

        # If batch has more than one circuit, return a list of ids
        if len(my_task.tasks) > 1:
            return [t.id for t in my_task.tasks]
        else:
            return my_task.tasks[0].id

    def job_status(self, job_id):
        """ Return job information corresponding to the input job id

        Args:
            job_id (str): string representing the job id

        Returns:
            enum | str : status response from the native API
        """
        return self.jobs[job_id].state()

    def job_results(self, job_id):
        """ Blocking call requesting job results.

        Args:
            job_id (str): string representing the job id

        Returns:
            dict: histogram of measurements
        """

        from braket.aws import AwsQuantumTask

        # Retrieve job object
        if job_id in self.jobs:
            job = self.jobs[job_id]
        else:
            job = AwsQuantumTask(arn=job_id)

        # Check status of job, raise error if results cannot be retrieved
        if self.job_status(job_id) in {'CANCELLED', 'FAILED'}:
            print(f"Job {job_id} was cancelled or failed, and no results can be retrieved.")
            return None

        # Retrieve results, update job dictionary.
        result = job.result()
        self.jobs_results[job_id] = result
        return result.measurement_probabilities if result else None

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

    @staticmethod
    def get_backend_info():
        """ Wrapper method to cut down the information returned by AWS SDK and provide a consistent interface for our code.

        Returns:
            Dictionary containing device information in the format:
                {
                    arn: {
                        provider: <provider>,
                        price: <price>,
                        unit: <unit>,
                    }
                }
        """

        return {device.arn: {
            "provider_name": device.provider_name,
            "price": device.properties.service.deviceCost.price,
            "unit": device.properties.service.deviceCost.unit}
            for device in refresh_available_braket_devices()}
