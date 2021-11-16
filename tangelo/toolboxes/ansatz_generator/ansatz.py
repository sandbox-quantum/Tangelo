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

"""This module defines the ansatz abstract class, providing the foundation to
implement variational ansatz circuits.
"""

import abc


class Ansatz(abc.ABC):
    """Base class for all Ansatz. Derived/children classes wirtten by users and
    developers must implement the following abstract methods.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def set_var_params(self):
        """Initialize variational parameters as zeros, random numbers, MP2, or
        any insightful values. Impacts the convergence of variational
        algorithms.
        """
        pass

    @abc.abstractmethod
    def prepare_reference_state(self):
        """Return circuit preparing the desired reference wavefunction (HF,
        multi-reference state...).
        """
        pass

    @abc.abstractmethod
    def build_circuit(self):
        """Build and return the quantum circuit implementing the ansatz."""
        pass

    @abc.abstractmethod
    def update_var_params(self):
        """Update value of variational parameters in the already-built ansatz
        circuit member.
        """
        pass
