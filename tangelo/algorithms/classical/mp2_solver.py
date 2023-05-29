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

"""Class performing electronic structure calculation employing the Moller-Plesset perturbation theory (MP2) method.
"""

from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.helpers.utils import is_package_installed


class MP2Solver(ElectronicStructureSolver):
    """Uses the Second-order Moller-Plesset perturbation theory (MP2) method to solve the electronic structure problem,
    through pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        mp2_fragment (pyscf.mp.MP2): The coupled-cluster object.
        mean_field (pyscf.scf.RHF): The mean field of the molecule.
        frozen (list or int): Frozen molecular orbitals.
    """

    def __init__(self, molecule):
        if not is_package_installed("pyscf"):
            raise ModuleNotFoundError(f"The pyscf package is not available and is required by {self.__class__.__name__}.")
        from pyscf import mp

        if molecule.uhf:
            raise NotImplementedError(f"SecondQuantizedMolecule that use UHF are not currently supported in {self.__class__.__name__}. Use CCSDSolver")

        self.mp = mp
        self.mp2_fragment = None

        self.spin = molecule.spin

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos
        self.uhf = molecule.uhf

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: MP2 energy.
        """
        # Execute MP2 calculation
        self.mp2_fragment = self.mp.MP2(self.mean_field, frozen=self.frozen)
        self.mp2_fragment.verbose = 0
        _, self.mp2_t2 = self.mp2_fragment.kernel()

        total_energy = self.mp2_fragment.e_tot

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if MP2 is performed
        if self.mp2_fragment is None:
            raise RuntimeError("MP2Solver: Cannot retrieve RDM. Please run the 'simulate' method first")
        if self.frozen is not None:
            raise RuntimeError("MP2Solver: RDM calculation is not implemented with frozen orbitals.")

        one_rdm = self.mp2_fragment.make_rdm1()
        two_rdm = self.mp2_fragment.make_rdm2()

        return one_rdm, two_rdm
