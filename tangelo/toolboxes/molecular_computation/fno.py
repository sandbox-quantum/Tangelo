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

"""Module containing datastructures for interfacing with the Frozen Natural
Orbitals (FNOs), to automatically truncate the virtual orbital space.

Reference: arXiv:2002.07901
"""

import warnings

import numpy as np

from tangelo.algorithms.classical import MP2Solver


class FNO:
    """Class to interface with the Frozen Natural Orbitals protocol, that aims
    at reducing the computational cost of a molecular problem by truncating
    the virtual space. In general, the virtual orbitals are ranked according to
    their MP2 occupancies, and selected with a given threshold. They are also
    transformed to the FNO basis, using the eigenvectors of the MP2
    virtual-virtual density matrix.

    Attributes:
        sqmol (SecondQuantizedMolecule): Self-explanatory.
        uhf (bool): Flag indicating the type of mean field used.
        n_mos (int): Number of molecular orbitals.
        fock_ao (np.array): Fock matrix in atomic orbital form.
        frozen_occupied (list): List of indices of frozen occupied orbitals.
        threshold (float or list): Threshold(s) for FNO occupancy.
    """

    def __init__(self, sqmol, threshold):
        """Initialization of the FNO class instance.

        Checks for frozen virtual orbitals and warns if they are set, as they
        might be overwritten.

        Args:
            sqmol (SecondQuantizedMolecule): The SecondQuantizedMolecule
                object.
            threshold (float or list): Threshold(s) for FNO occupancy.
        """

        # Check if there are frozen virtual orbitals. Print warning that this
        # setting will be overwritten.
        if sqmol.frozen_virtual and sqmol.frozen_virtual != [[], []]:
            warnings.warn(f"The frozen orbitals indices will be overwritten in {self.__class__.__name__}.", RuntimeWarning)

        self.sqmol = sqmol

        # Molecular properties useful for the FNO class methods.
        self.uhf = sqmol.uhf
        self.n_mos = self.sqmol.n_mos
        self.fock_ao = self.sqmol.mean_field.get_fock()
        self.frozen_occupied = self.sqmol.frozen_occupied

        # Verify threshold format.
        if not isinstance(threshold, (int, float)) or not 0. < threshold <= 1.:
            raise ValueError(f"The threshold {threshold} is invalid, the %FNO occupancy threshold must be within 0 and 1.")

        self.threshold = threshold

        if self.uhf:
            self.n_occupied = [len(x+y) for x, y in zip(self.sqmol.frozen_occupied, self.sqmol.active_occupied)]
            self._compute_ump2_densities()
        elif self.sqmol.spin != 0:
            raise NotImplementedError("ROHF is not supported for FNO. Please use UHF for open-shell systems.")
        else:
            self.n_occupied = len(self.sqmol.frozen_occupied + self.sqmol.active_occupied)
            self._compute_rmp2_densities()

        self.compute_fno(self.threshold)

    @property
    def fermionic_hamiltonian(self):
        """Property that returns a FNO fermionic hamiltonian object, with the
        truncated active space and updated MO coefficients.

        Returns:
            FermionOperator: Self-explanatory.
        """
        frozen_orbitals = self.get_frozen_indices()
        sqmol_updated = self.sqmol.freeze_mos(frozen_orbitals, inplace=False)
        return sqmol_updated._get_fermionic_hamiltonian(self.mo_coeff)

    def compute_fno(self, threshold):
        """Method to compute and truncate the FNO orbitals. It calls
        the `_compute_rfno` or the `_compute_ufno`method, whichever is
        appropriate.
        """
        self._compute_ufno(threshold) if self.uhf else self._compute_rfno(threshold)

    def get_frozen_indices(self):
        """Method to determine the indices of the frozen orbitals, and it calls
        the `_get_restricted_frozen_indices` or the `_get_unrestricted_frozen_indices`
        method, whether is appropriate.
        """
        return self._get_unrestricted_frozen_indices() if self.uhf else self._get_restricted_frozen_indices()

    def _get_restricted_frozen_indices(self):
        """Method to determine the indices of the frozen orbitals in a
        restricted calculation.

        Returns:
            list of int: List containing the frozen orbital indices for the
                molecular orbitals.
        """

        # All are set to False.
        moidx = np.zeros(self.n_mos, dtype=bool)

        # True starting from first index until an active orbital index.
        active_virt_fno = self.n_occupied + self.n_active_virt_fno
        moidx[: active_virt_fno] = True

        # Set frozen occupied orbital to False.
        moidx[self.frozen_occupied] = False

        # Obtain the frozen indices.
        frozen_indices = np.where(moidx == 0)[0].tolist()

        return frozen_indices

    def _get_unrestricted_frozen_indices(self):
        """Method to determine the indices of the frozen orbitals in an
        unrestricted calculation.

        Returns:
            list of int list: List containing the frozen orbital indices for
                alpha and beta spins.
        """

        frozen_indices = [None] * 2
        for is_beta_spin, n_active_virt_fno in enumerate(self.n_active_virt_fno):

            # All are set to False.
            moidx = np.zeros(self.n_mos, dtype=bool)

            # True starting from first index until an active orbital index.
            active_virt_fno = self.n_occupied[is_beta_spin] + n_active_virt_fno
            moidx[: active_virt_fno] = True

            # Set frozen occupied orbital to False.
            moidx[self.frozen_occupied[is_beta_spin]] = False

            # Obtain the frozen indices
            frozen_indices[is_beta_spin] = np.where(moidx == 0)[0].tolist()

        return frozen_indices

    def _compute_mp2(self):
        """"Method to compute the RMP2 or UMP2 classical solution.

        Returns:
            t2 (np.array or tuple of np.array): T2 amplitudes, in the pyscf
                format. The exact format depends if the calculation has been
                from a RHF or a UHF mean field.
        """

        mp2 = MP2Solver(self.sqmol)
        mp2.simulate()

        return mp2.solver.mp2_t2

    def _compute_rmp2_densities(self):
        """Method to computes the restricted MP2 densities for further
        consideration. They are diagonalized, and the eigenvalues and
        eigenvectors are used to rank and transform the virtual block of the
        molecular orbitals.
        """

        # Compute the RMP2 solution.
        t2 = self._compute_mp2()

        # Compute density matrix of the virtual-virtual block.
        rho_virtvirt = self._compute_virt_virt_rmp2_density(t2)

        self.fno_occ, self.unitary = self.diagonalize_and_reorder(rho_virtvirt, reorder=True)

    def _compute_virt_virt_rmp2_density(self, t2):
        """Method to compute the virtual-virtual density matrices for a
        restricted MP2 calculation.

        Args:
            t2 (np.array): Array containing T2 amplitudes, in the pyscf format.

        Returns:
            np.array: Virtual-virtual density matrix.
        """

        dvv = np.zeros(t2.shape[2:4], dtype=t2.dtype)

        for i in range(t2.shape[0]):
            t2i = t2[i]
            l2i = t2i.conj()
            dvv += np.einsum("jca,jcb->ba", l2i, t2i) * 2 - np.einsum("jca,jbc->ba", l2i, t2i)

        return dvv + dvv.conj().T

    def _compute_ump2_densities(self):
        """Method to computes the unrestricted MP2 densities for further
        consideration. They are diagonalized, and the eigenvalues and
        eigenvectors are used to rank and transform the virtual block of the
        molecular orbitals.
        """

        # Compute the UMP2 solution.
        t2 = self._compute_mp2()

        # Compute density matrix of the virtual-virtual block.
        rho_virtvirt = self._compute_virt_virt_ump2_density(t2)

        self.fno_occ = [None] * 2
        self.unitary = [None] * 2
        for is_beta_spin, rho in enumerate(rho_virtvirt):
            occ, unitary = self.diagonalize_and_reorder(rho, reorder=True)
            self.fno_occ[is_beta_spin] = occ
            self.unitary[is_beta_spin] = unitary

    def _compute_virt_virt_ump2_density(self, t2):
        """Method to compute the virtual-virtual density matrices for an
        unrestricted MP2 calculation.

        Args:
            t2 (tuple of arrays): Array containing T2 amplitudes, in the
                alpha-alpha, alpha-beta and beta-beta format (like pyscf).

        Returns:
            tuple: Virtual-virtual density matrices for alpha and beta spins.
        """

        # Unpack the T2 amplitudes.
        t2aa, t2ab, t2bb = t2

        dvva = np.einsum("mnae,mnbe->ba", t2aa.conj(), t2aa) * .5
        dvva += np.einsum("mnae,mnbe->ba", t2ab.conj(), t2ab)

        dvvb = np.einsum("mnae,mnbe->ba", t2bb.conj(), t2bb) * .5
        dvvb += np.einsum("mnea,mneb->ba", t2ab.conj(), t2ab)

        return (dvva + dvva.conj().T, dvvb + dvvb.conj().T)

    def _compute_rfno(self, threshold):
        """Method to apply the Frozen Natural Orbitals (FNOs) based on a
        specified threshold. This method deals with the restricted mean-field
        formalism.

        Args:
            threshold (int or float): Threshold for FNO occupancy.
        """

        self.n_active_virt_fno = self.get_number_of_fnos_from_frac_occupancies(
            self.fno_occ, threshold)

        self.mo_coeff = self._compute_mo_coeff(
            self.n_occupied,
            self.n_active_virt_fno,
            self.sqmol.mo_coeff,
            self.unitary,
            self.fock_ao
        )

    def _compute_ufno(self, threshold):
        """Method to apply the Frozen Natural Orbitals (FNOs) based on a
        specified threshold. This method deals with the unrestricted mean-field
        formalism.

        Args:
            threshold (list of int or float): List of thresholds for
                FNO occupancy. Entries are for the alpha and beta spinorbitals,
                respectively.
        """

        self.mo_coeff = [None] * 2
        self.n_active_virt_fno = [None] * 2

        # 0 is for alpha, 1 is for beta.
        for is_beta_spin in range(2):

            n_active_virt_fno = self.get_number_of_fnos_from_frac_occupancies(
                self.fno_occ[is_beta_spin], threshold)

            self.n_active_virt_fno[is_beta_spin] = n_active_virt_fno

            self.mo_coeff[is_beta_spin] = self._compute_mo_coeff(
                self.n_occupied[is_beta_spin],
                n_active_virt_fno,
                self.sqmol.mo_coeff[is_beta_spin],
                self.unitary[is_beta_spin],
                self.fock_ao[is_beta_spin]
            )

    def _compute_mo_coeff(self, n_occ, n_active_virt_fnos, mo_coeff_scf, unitary, fock_ao):
        """Method to compute the Molecular Orbital (MO) coefficients based on
        provided parameters.

        It transforms the MO coefficients into Frozen Natural Orbitals
        (FNOs). The high-level steps involves slicing the virtual block from
        'mo_coeff_scf', transforming it using the provided 'unitary', and
        slicing again the active part of virtual orbitals to obtain FNOs.

        The Fock matrix is then recanonicalized using the resulting
        coefficients. Diagonalizing the Fock matrix in the MO basis is
        performed to get the full set of MO coefficients.

        Args:
            n_occ (int): Number of occupied orbitals.
            n_active_virt_fnos (int): Number of active virtual in the Frozen
                Natural Orbitals basis.
            mo_coeff_scf (np.array): Molecular Orbital coefficients from the
                self-consistent field (SCF) calculation.
            unitary (np.array): Unitary matrix corresponding to the sorted
                eigenvectors of the MP2 densities.
            fock_ao (np.array): Fock matrix in atomic orbital (AO) form.

        Returns:
            np.array: Updated Molecular Orbital coefficients based on the
                computed FNOs.
        """

        # Transform the MO coefficients in FNO.
        mo_coeff_virtual = mo_coeff_scf[:, n_occ:]
        mo_coeff_virtual_fno = mo_coeff_virtual @ unitary
        mo_coeff_fno = mo_coeff_virtual_fno[:, :n_active_virt_fnos]

        # Obtain Fock matrix in MO form.
        fock_mo = mo_coeff_fno.T @ fock_ao @ mo_coeff_fno

        # Diagonalize the new Fock matrix and apply the transformation.
        _,  unitary_sc = self.diagonalize_and_reorder(fock_mo, reorder=False)
        mo_coeff_new = mo_coeff_fno @ unitary_sc

        n_active_space = n_occ + n_active_virt_fnos

        # Update the mo_coeff
        mo_coeff_occ = mo_coeff_scf[:, : n_occ]
        mo_coeff_sc = mo_coeff_new[:, :]
        mo_coeff_vrt = mo_coeff_scf[:, n_active_space:]
        mo_coeff = np.hstack((mo_coeff_occ, mo_coeff_sc, mo_coeff_vrt))

        return mo_coeff

    @staticmethod
    def diagonalize_and_reorder(m, reorder=True):
        """Method to diagonalize a matrix and reorder the eigenvalues and
        eigenvectors based on occupations.

        Args:
            m (np.array): The matrix to be diagonalized.
            reorder (bool): Flag indicating whether to reorder the eigenvalues
                and eigenvectors based on occupations. Defaults to True.

        Returns:
            tuple: A tuple containing the reordered eigenvalues and the
                corresponding eigenvectors. The eigenvalues represent
                occupations, and the eigenvectors represent rotation operators.
        """

        # Obtain natural orbitals.
        eigen_vals, eigen_vecs = np.linalg.eigh(m)

        # Sort according to the occupations.
        order = -1 if reorder else 1
        idx = eigen_vals.argsort()[::order]
        occupations = eigen_vals[idx]
        rotation_op = eigen_vecs[:, idx]

        return occupations, rotation_op

    @staticmethod
    def get_number_of_fnos_from_frac_occupancies(fno_occ, threshold_frac_occ):
        """Method to calculate the number of Frozen Natural Orbitals (FNOs)
        to consider, based on fractional occupancies.

        Args:
            fno_occ (np.array): Array containing fractional occupancies of FNOs.
            threshold_frac_occ (float): Threshold value for fractional occupancy.

        Returns:
            int: Number of FNOs determined by the cumulative sum of fractional
                occupancies satisfying the provided threshold.
        """

        fno_occ_cumul_frac_occ = np.cumsum(fno_occ) / np.sum(fno_occ)
        where_threshold = np.where(fno_occ_cumul_frac_occ <= threshold_frac_occ, 1, 0)

        # +1 to be equivalent to the old function it is based on.
        number_of_fnos = np.sum(where_threshold) + 1

        return number_of_fnos
