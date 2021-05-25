""" Class performing electronic structure calculation employing the semi-empirical methods.

Here are semi-empirical methods implemented:
- MINDO3
"""

from pyscf.semiempirical import mindo3

from .electronic_structure_solver import ElectronicStructureSolver


class MINDO3Solver(ElectronicStructureSolver):
    """Uses the MINDO3 method to solve the electronic structure problem,
    through pyscf. Only the restricted (RMINDO3) flavor is implemented.

    Refs:
    R. C. Bingham, M. J. Dewar, D. H. Lo, J. Am. Chem. Soc., 97, 1285 (1975)
    D. F. Lewis, Chem. Rev. 86, 1111 (1986).
    """

    def __init__(self):
        pass

    def simulate(self, molecule):
        """Perform the simulation (energy calculation) for the molecule.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.

        Returns:
            total_energy (float): RMINDO3 energy
        """

        solver = mindo3.RMINDO3(molecule).run()
        total_energy = solver.e_tot

        return total_energy

    def get_rdm(self):
        """Method must be defined (ElectronicStructureSolver). For semi-empirical
        methods, it is not relevant nor defined.
        """

        raise NotImplementedError("Method get_rdm is not relevant for semi-empirical methods.")
