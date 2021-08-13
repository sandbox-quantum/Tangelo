"""Docstring."""

from pyscf import scf

from .electronic_structure_solver import ElectronicStructureSolver


class RHFSolver(ElectronicStructureSolver):
    """Docstring. """

    def __init__(self, molecule, basis_set="sto-3g"):
        self.mean_field = scf.RHF(molecule.to_pyscf(basis_set))
        self.mean_field.verbose = 0

    def simulate(self):
        """Docstring. """

        self.mean_field.scf()

        return self.mean_field.e_tot

    def get_rdm(self):
        pass
