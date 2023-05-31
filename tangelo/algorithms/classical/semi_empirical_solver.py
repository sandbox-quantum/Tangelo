# Copyright 2023 Good Chemistry Company.
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

"""Class performing electronic structure calculation employing the
semi-empirical methods. At first, semi-empirical methods are ways of computing
the total energy of a molecule in a very fast way to optimize its geometry.
Those methods are not ab initio as they employ empirical parameters, as stated
in the name "semi-empirical". They are in fact related to simplified
Hartree-Fock versions with empirical corrections. Differences between them come
from the process chosen to compute the empirical parameters. For example, MINDO3
inventors used atomization energies to fit their mathematical models.

They have been introduced in this package for the purpose of computing an
environment energy and inducing constraints on atomic positions. As stand-alone
solvers, they are however a poor choice, as they do not provide an accurate
approximation of energies.

Here are the semi-empirical method(s) implemented:
    - MINDO3
"""

from tangelo.helpers.utils import is_package_installed
from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.toolboxes.molecular_computation.integral_solver_pyscf import mol_to_pyscf


class MINDO3Solver(ElectronicStructureSolver):
    """Uses the MINDO3 method to solve the electronic structure problem, through
    pyscf. Only the restricted (RMINDO3) flavor is implemented.

    Args:
        molecule (Molecule or SecondQuantizedMolecule): The molecule to
            simulate.

    Refs:
        - R. C. Bingham, M. J. Dewar, D. H. Lo, J. Am. Chem. Soc., 97, 1285
            (1975).
        - D. F. Lewis, Chem. Rev. 86, 1111 (1986).
    """

    def __init__(self, molecule):
        if not is_package_installed("pyscf.semiempirical"):
            raise ModuleNotFoundError(f"The pyscf.semiempirical module is not available and is required by {self.__class__.__name__}.")

        self.molecule = molecule

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: RMINDO3 energy.
        """
        from pyscf.semiempirical import mindo3

        solver = mindo3.RMINDO3(mol_to_pyscf(self.molecule)).run(verbose=0)
        total_energy = solver.e_tot

        return total_energy

    def get_rdm(self):
        """Method must be defined (ElectronicStructureSolver). For
        semi-empirical methods, it is not relevant nor defined.
        """

        raise NotImplementedError("Method get_rdm is not relevant for semi-empirical methods.")
