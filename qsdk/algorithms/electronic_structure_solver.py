"""Abstract class defining a common interface to all electronic structure
solvers, in order to have consistency and ensure compatibility with higher-level
workflows that can use them, such as problem decomposition methods.
"""

import abc


class ElectronicStructureSolver(abc.ABC):
    """Sets interface for objects that perform energy estimation of a molecule.

    Specifics vary between concrete implementations, but common to all of them
    is that after `simulate` is called, the right internal state is set for the
    class so that `get_rdm` can return reduced density matrices for the last run
    simulation.
    """

    def __init__(self, molecule):
        """Performs the simulation for a molecule. The mean field is calculated
        automatically.

        Args:
            molecule (SecondQuantizedMolecule): The molecule on which to perform the
                simulation.
        """
        pass

    @abc.abstractmethod
    def simulate(self):
        """Performs the simulation for a molecule."""
        pass

    @abc.abstractmethod
    def get_rdm(self):
        """Returns the RDM for the previous simulation.
        In a concrete implementation, the `simulate` function would set the
        necessary internal state for the class so that this function can return
        the reduced density matrix.

        Returns:
            (numpy.array, numpy.array): The one- and two-particle RDMs
            (float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """
        pass
