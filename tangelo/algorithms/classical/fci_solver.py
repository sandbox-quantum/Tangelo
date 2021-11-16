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

"""Define electronic structure solver employing the full configuration
interaction (CI) method.
"""

from pyscf import ao2mo, fci, mcscf

from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver


class FCISolver(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem,
    through pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        mean_field (pyscf.scf): Mean field object.
    """

    def __init__(self, molecule):

        self.ci = None
        self.norb = molecule.n_active_mos
        self.nelec = molecule.n_active_electrons
        self.spin = molecule.spin
        self.n_alpha = self.nelec//2 + self.spin//2 + (self.nelec % 2)
        self.n_beta = self.nelec//2 - self.spin//2

        # Need to use a CAS method if frozen orbitals are defined
        if molecule.frozen_mos is not None:
            # Generate CAS space with given frozen_mos, then use pyscf functionality to
            # obtain effective Hamiltonian with frozen orbtials excluded from the CI space.
            self.cas = True
            self.cassolver = mcscf.CASSCF(molecule.mean_field,
                                          molecule.n_active_mos,
                                          (self.n_alpha, self.n_beta),
                                          frozen=molecule.frozen_orbitals)
            self.h1e_cas, self.ecore = self.cassolver.get_h1eff()
            self.h2e_cas = self.cassolver.get_h2eff()
            # Initialize the FCI solver that will use the effective Hamiltonian generated from CAS
            self.cisolver = fci.direct_spin1.FCI()
        else:
            self.cas = False
            if self.spin == 0:
                self.cisolver = fci.direct_spin0.FCI(molecule.to_pyscf(molecule.basis))
            else:
                self.cisolver = fci.direct_spin1.FCI()

        self.cisolver.verbose = 0
        self.mean_field = molecule.mean_field

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total FCI energy.
        """

        if self.cas:  # Use previously generated effective Hamiltonian to obtain FCI solution
            energy, self.ci = self.cisolver.kernel(self.h1e_cas,
                                                   self.h2e_cas,
                                                   self.norb,
                                                   (self.n_alpha, self.n_beta),
                                                   ecore=self.ecore)
        else:  # Generate full Hamiltonian and obtain FCI solution.
            h1 = self.mean_field.mo_coeff.T @ self.mean_field.get_hcore() @ self.mean_field.mo_coeff

            twoint = self.mean_field._eri

            eri = ao2mo.restore(8, twoint, self.norb)
            eri = ao2mo.incore.full(eri, self.mean_field.mo_coeff)
            eri = ao2mo.restore(1, eri, self.norb)

            ecore = self.mean_field.energy_nuc()

            if self.spin == 0:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], self.nelec, ecore=ecore)
            else:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], (self.n_alpha, self.n_beta), ecore=ecore)

        return energy

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If method "simulate" hasn't been run.
        """

        # Check if Full CI is performed
        if self.ci is None:
            raise RuntimeError("FCISolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        if self.cas:
            one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))
        else:
            if self.spin == 0:
                one_rdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
                two_rdm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)
            else:
                one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))

        return one_rdm, two_rdm
