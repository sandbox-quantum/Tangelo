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

from itertools import combinations, product
from math import ceil

from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.helpers.utils import is_package_installed
from tangelo.toolboxes.ansatz_generator._unitary_cc_openshell import uccsd_openshell_get_packed_amplitudes


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
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")
        from pyscf import mp

        self.mp = mp
        self.mp2_fragment = None

        self.spin = molecule.spin

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos
        self.uhf = molecule.uhf

        # Define variables used to transform the MP2 parameters into an ordered
        # list of parameters with single and double excitations.
        if self.spin != 0 or self.uhf:
            self.n_alpha, self.n_beta = molecule.n_active_ab_electrons
            self.n_active_moa, self.n_active_mob = molecule.n_active_mos if self.uhf else (molecule.n_active_mos,)*2
        else:
            self.n_occupied = ceil(molecule.n_active_electrons / 2)
            self.n_virtual = molecule.n_active_mos - self.n_occupied

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: MP2 energy.
        """

        # Execute MP2 calculation
        if self.uhf:
            self.mp2_fragment = self.mp.UMP2(self.mean_field, frozen=self.frozen)
        else:
            self.mp2_fragment = self.mp.RMP2(self.mean_field, frozen=self.frozen)

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

        # Check if MP2 has been performed
        if self.mp2_fragment is None:
            raise RuntimeError(f"{self.__class__.name}: Cannot retrieve RDM. Please run the 'simulate' method first")
        if self.frozen is not None:
            raise RuntimeError(f"{self.__class__.name}: RDM calculation is not implemented with frozen orbitals.")

        one_rdm = self.mp2_fragment.make_rdm1()
        two_rdm = self.mp2_fragment.make_rdm2()

        return one_rdm, two_rdm

    def get_mp2_amplitudes(self):
        """Compute the double amplitudes from the MP2 perturbative method, and
        then reorder the elements into a dense list. The single (T1) amplitudes
        are set to a small non-zero value. The ordering is single, double
        (diagonal), double (non-diagonal).

        Returns:
            list of float: The electronic excitation amplitudes.
        """

        # Check if MP2 has been performed.
        if self.mp2_fragment is None:
            raise RuntimeError(f"{self.__class__.name}: Cannot retrieve MP2 parameters. Please run the 'simulate' method first")

        if self.spin != 0 or self.uhf:
            # Reorder the T2 amplitudes in a dense list.
            mp2_params = uccsd_openshell_get_packed_amplitudes(
                self.mp2_t2[0],  # aa
                self.mp2_t2[2],  # bb
                self.mp2_t2[1],  # ab
                self.n_alpha,
                self.n_beta,
                self.n_active_moa,
                self.n_active_mob
            )
        else:
            # Get singles amplitude. Just get "up" amplitude, since "down" should be the same
            singles = [2.e-5] * (self.n_virtual * self.n_occupied)

            # Get singles and doubles amplitudes associated with one spatial occupied-virtual pair
            doubles_1 = [-self.mp2_t2[q, q, p, p]/2. if (abs(-self.mp2_t2[q, q, p, p]/2.) > 1e-15) else 0.
                        for p, q in product(range(self.n_virtual), range(self.n_occupied))]

            # Get doubles amplitudes associated with two spatial occupied-virtual pairs
            doubles_2 = [-self.mp2_t2[q, s, p, r] for (p, q), (r, s)
                        in combinations(product(range(self.n_virtual), range(self.n_occupied)), 2)]

            mp2_params = singles + doubles_1 + doubles_2

        return mp2_params
