# Copyright 2021 Good Chemistry Company.
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

"""Abstract parent class encapsulating basic features of compute services for
quantum circuit simulation.
"""

import abc


class QpuConnection(abc.ABC):
    """Abstract class encapsulating login/authentication setup and job
    submission and management.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def job_submit(self):
        """Submit a job to the compute services."""
        pass

    @abc.abstractmethod
    def job_get_info(self, job_id):
        """Retrieve information about a previously submitted job, through its
        job id.
        """
        pass

    @abc.abstractmethod
    def job_get_results(self, job_id):
        """Retrieve the results of previously submitted job, through its job id."""
        pass

    @abc.abstractmethod
    def job_cancel(self, job_id):
        """Attempt to cancel a previously submitted job, through its job id."""
        pass
