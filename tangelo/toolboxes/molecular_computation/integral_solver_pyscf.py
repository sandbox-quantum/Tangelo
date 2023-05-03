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

import numpy as np

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


class IntegralSolver_pyscf(IntegralSolver):
    """Instantiate Electronic Structure integration"""
    def __init__(self):
        from pyscf import gto, lib, scf, symm, ao2mo
        self.gto = gto
        self.lib = lib
        self.scf = scf
        self.symm = symm
        self.ao2mo = ao2mo

    def set_basic_data(self, tmol):
        _ = self.to_pyscf(tmol)

    def to_pyscf(self, tmol, basis="CRENBL", symmetry=False, ecp=None):
        """Method to return a pyscf.gto.Mole object.

        Args:
            basis (string): Basis set.
            symmetry (bool): Flag to turn symmetry on
            ecp (dict): Dictionary with ecp definition for each atom e.g. {"Cu": "crenbl"}

        Returns:
            pyscf.gto.Mole: PySCF compatible object.
        """

        mol = self.gto.Mole(atom=tmol.xyz)
        mol.basis = basis
        mol.charge = tmol.q
        mol.spin = tmol.spin
        mol.symmetry = symmetry
        mol.ecp = ecp if ecp else dict()
        mol.build()

        tmol.xyz = list()
        for sym, xyz in mol._atom:
            tmol.xyz += [tuple([sym, tuple([x*self.lib.parameters.BOHR for x in xyz])])]

        tmol.n_atoms = mol.natm
        tmol.n_electrons = mol.nelectron

        return mol

    def compute_mean_field(self, tmol):
        """Computes the mean-field for the molecule. Depending on the molecule
        spin, it does a restricted or a restriction open-shell Hartree-Fock
        calculation.

        It is also used for defining attributes related to the mean-field
        (mf_energy, mo_energies, mo_occ, n_mos and n_sos).
        """

        molecule = self.to_pyscf(tmol, tmol.basis, tmol.symmetry, tmol.ecp)

        tmol.mean_field = self.scf.RHF(molecule) if not tmol.uhf else self.scf.UHF(molecule)
        tmol.mean_field.verbose = 0

        # Force broken symmetry for uhf calculation when spin is 0 as shown in
        # https://github.com/sunqm/pyscf/blob/master/examples/scf/32-break_spin_symm.py
        if tmol.uhf and tmol.spin == 0:
            dm_alpha, dm_beta = tmol.mean_field.get_init_guess()
            dm_beta[:1, :] = 0
            dm = (dm_alpha, dm_beta)
            tmol.mean_field.kernel(dm)
        else:
            tmol.mean_field.kernel()

        tmol.mean_field.analyze()
        if not tmol.mean_field.converged:
            raise ValueError("Hartree-Fock calculation did not converge")

        if tmol.symmetry:
            tmol.mo_symm_ids = list(self.symm.label_orb_symm(tmol.mean_field.mol, tmol.mean_field.mol.irrep_id,
                                                             tmol.mean_field.mol.symm_orb, tmol.mean_field.mo_coeff))
            irrep_map = {i: s for s, i in zip(molecule.irrep_name, molecule.irrep_id)}
            tmol.mo_symm_labels = [irrep_map[i] for i in tmol.mo_symm_ids]
        else:
            tmol.mo_symm_ids = None
            tmol.mo_symm_labels = None

        tmol.mf_energy = tmol.mean_field.e_tot
        tmol.mo_energies = tmol.mean_field.mo_energy
        tmol.mo_occ = tmol.mean_field.mo_occ

        tmol.n_mos = molecule.nao_nr()
        tmol.n_sos = 2*tmol.n_mos

        self.mo_coeff = tmol.mean_field.mo_coeff

    def get_integrals(self, tmol, mo_coeff=None):
        """Computes core constant, one_body, and two-body coefficients for a given active space and mo_coeff
        For UHF
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.
            consider_frozen (bool): If True, the frozen orbitals are folded into the one_body and core constant terms.

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """

        # Pyscf molecule to get integrals.
        pyscf_mol = self.to_pyscf(tmol, tmol.basis, tmol.symmetry, tmol.ecp)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if tmol.uhf:
            one_body, two_body = self.compute_uhf_integrals(tmol, mo_coeff)
            return float(pyscf_mol.energy_nuc()), one_body, two_body

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(pyscf_mol.energy_nuc())

        # get_hcore is equivalent to int1e_kin + int1e_nuc.
        one_electron_integrals = mo_coeff.T @ tmol.mean_field.get_hcore() @ mo_coeff

        # Getting 2-body integrals in atomic and converting to molecular basis.
        two_electron_integrals = self.ao2mo.kernel(pyscf_mol.intor("int2e"), mo_coeff)
        two_electron_integrals = self.ao2mo.restore(1, two_electron_integrals, len(mo_coeff))

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # made before performing the truncation for frozen orbitals.
        two_electron_integrals = two_electron_integrals.transpose(0, 2, 3, 1)

        return core_constant, one_electron_integrals, two_electron_integrals

    def compute_uhf_integrals(self, tmol, mo_coeff):
        """Compute 1-electron and 2-electron integrals
        The return is formatted as
        [numpy.ndarray]*2 numpy array h_{pq} for alpha and beta blocks
        [numpy.ndarray]*3 numpy array storing h_{pqrs} for alpha-alpha, alpha-beta, beta-beta blocks

        Args:
            List[array]: The molecular orbital coefficients for both spins [alpha, beta]

        Returns:
            List[array], List[array]: One and two body integrals
        """
        # step 1 : find nao, nmo (atomic orbitals & molecular orbitals)

        # molecular orbitals (alpha and beta will be the same)
        # Lets take alpha blocks to find the shape and things

        # molecular orbitals
        nmo = tmol.nmo = mo_coeff[0].shape[1]
        # atomic orbitals
        nao = tmol.nao = mo_coeff[0].shape[0]

        # step 2 : obtain Hcore Hamiltonian in atomic orbitals basis
        hcore = tmol.mean_field.get_hcore()

        # step 3 : obatin two-electron integral in atomic basis
        eri = self.ao2mo.restore(8, tmol.mean_field._eri, nao)

        # step 4 : create the placeholder for the matrices
        # one-electron matrix (alpha, beta)
        hpq = []

        # step 5 : do the mo transformation
        # step the mo coeff alpha and beta
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]

        # mo transform the hcore
        hpq.append(mo_a.T.dot(hcore).dot(mo_a))
        hpq.append(mo_b.T.dot(hcore).dot(mo_b))

        # mo transform the two-electron integrals
        eri_a = self.ao2mo.incore.full(eri, mo_a)
        eri_b = self.ao2mo.incore.full(eri, mo_b)
        eri_ba = self.ao2mo.incore.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False)

        # Change the format of integrals (full)
        eri_a = self.ao2mo.restore(1, eri_a, nmo)
        eri_b = self.ao2mo.restore(1, eri_b, nmo)
        eri_ba = eri_ba.reshape(nmo, nmo, nmo, nmo)

        # # convert this into the order OpenFemion like to receive
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Gpqrs has alpha, alphaBeta, Beta blocks
        Gpqrs = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        return hpq, Gpqrs
