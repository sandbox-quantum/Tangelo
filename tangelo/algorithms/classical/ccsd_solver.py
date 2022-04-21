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

"""Class performing electronic structure calculation employing the CCSD method.
"""

import numpy as np

from pyscf import cc, lib
import pyscf.cc.ccsd_rdm as ccsd_rdm
import pyscf.cc.uccsd_rdm as uccsd_rdm

from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver


class CCSDSolver(ElectronicStructureSolver):
    """Uses the CCSD method to solve the electronic structure problem, through
    pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
        mean_field (pyscf.scf.RHF): The mean field of the molecule.
        frozen (list or int): Frozen molecular orbitals.
    """

    def __init__(self, molecule):
        self.cc_fragment = None

        self.spin = molecule.spin

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: CCSD energy.
        """
        # Execute CCSD calculation
        self.cc_fragment = cc.CCSD(self.mean_field, frozen=self.frozen)
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7

        correlation_energy, _, _ = self.cc_fragment.ccsd()
        total_energy = self.mean_field.e_tot + correlation_energy

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices. The CCSD
        lambda equation will be solved for calculating the RDMs.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if CCSD calculation is performed
        if self.cc_fragment is None:
            raise RuntimeError("CCSDSolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        t1 = self.cc_fragment.t1
        t2 = self.cc_fragment.t2
        l1, l2 = self.cc_fragment.solve_lambda(t1, t2)

        f = lib.H5TmpFile()

        if self.spin == 0:
            d1 = ccsd_rdm._gamma1_intermediates(self.cc_fragment, t1, t2, l1, l2)
            d2 = ccsd_rdm._gamma2_outcore(self.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = ccsd_rdm._make_rdm1(self.cc_fragment, d1, with_frozen=False)
            two_rdm = ccsd_rdm._make_rdm2(self.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)
        else:
            d1 = uccsd_rdm._gamma1_intermediates(self.cc_fragment, t1, t2, l1, l2)
            d2 = uccsd_rdm._gamma2_outcore(self.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = uccsd_rdm._make_rdm1(self.cc_fragment, d1, with_frozen=False)
            two_rdm = uccsd_rdm._make_rdm2(self.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)

            one_rdm = np.sum(one_rdm, axis=0)
            two_rdm = np.sum((two_rdm[0], 2*two_rdm[1], two_rdm[2]), axis=0)

        return one_rdm, two_rdm
