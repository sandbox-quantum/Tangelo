""" Abstract parent class encapsulating basic features of compute services for quantum circuit simulation """

import abc


class QpuConnection(abc.ABC):
    """ Abstract class encapsulating login/authentication setup and job submission and management  """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def job_submit(self):
        """ Submit a job to the compute services """
        pass

    @abc.abstractmethod
    def job_status(self, job_id):
        """ Retrieve information about a previously submitted job, through its job id """
        pass

    @abc.abstractmethod
    def job_results(self, job_id):
        """ Retrieve the results of previously submitted job, through its job id """
        pass

    @abc.abstractmethod
    def job_cancel(self, job_id):
        """ Attempt to cancel a previously submitted job, through its job id """
        pass
