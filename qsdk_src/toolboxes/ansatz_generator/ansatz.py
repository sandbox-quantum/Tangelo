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
