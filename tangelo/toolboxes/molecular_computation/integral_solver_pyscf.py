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

import os

import numpy as np

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


def mol_to_pyscf(mol, basis="CRENBL", symmetry=False, ecp=None):
    """Method to return a pyscf.gto.Mole object.

    Args:
        sqmol (SecondQuantizedMolecule or Molecule): The molecule to export to a pyscf molecule.
        basis (string): Basis set.
        symmetry (bool): Flag to turn symmetry on
        ecp (dict): Dictionary with ecp definition for each atom e.g. {"Cu": "crenbl"}

    Returns:
        pyscf.gto.Mole: PySCF compatible object.
    """
    from pyscf import gto

    pymol = gto.Mole(atom=mol.xyz)
    pymol.basis = basis
    pymol.charge = mol.q
    pymol.spin = mol.spin
    pymol.symmetry = symmetry
    pymol.ecp = ecp if ecp else dict()
    pymol.build()

    return pymol


class IntegralSolverPySCF(IntegralSolver):
    """Electronic Structure integration for pyscf"""

    def __init__(self, chkfile=None):
        """Initialize the integral solver class for pyscf. A chkfile path can be
        provided.

        Regarding the chkfile, three scenarios are possible:
        - A chkfile path is provided, but the file doesn't exist: it creates
            a chkfile at the end of the SCF calculation.
        - A chkfile path is provided and a file already exists: the initial
            guess is taken from the chkfile and this file is updated at the end
            of the calculation.
        - No chkfile path is provided: The SCF initial guess stays the default
            one (minao). No chkfile is created.

        Args:
            chkfile (string): Path of the chkfile.
        """

        from pyscf import gto, lib, scf, symm, ao2mo
        self.gto = gto
        self.lib = lib
        self.scf = scf
        self.symm = symm
        self.ao2mo = ao2mo
        self.chkfile = chkfile

    def set_physical_data(self, mol):
        """Set molecular data that is independant of basis set in mol

        Modify mol variable:
            mol.xyz to (list): Nested array-like structure with elements and coordinates
                                            (ex:[ ["H", (0., 0., 0.)], ...]) in angstrom
        Add to mol:
            mol.n_electrons (int): Self-explanatory.
            mol.n_atoms (int): Self-explanatory.

        Args:
            mol (Molecule or SecondQuantizedMolecule): Class to add the other variables given populated.
                mol.xyz (in appropriate format for solver): Definition of molecular geometry.
                mol.q (float): Total charge.
                mol.spin (int): Absolute difference between alpha and beta electron number.
        """
        pymol = mol_to_pyscf(mol)
        mol.xyz = list()
        for sym, xyz in pymol._atom:
            mol.xyz += [tuple([sym, tuple([x*self.lib.parameters.BOHR for x in xyz])])]

        mol.n_atoms = pymol.natm
        mol.n_electrons = pymol.nelectron

    def compute_mean_field(self, sqmol):
        """Run a unrestricted/restricted (openshell-)Hartree-Fock calculation and modify/add the following
        variables to sqmol

        Modify sqmol variables.
            sqmol.mf_energy (float): Mean-field energy (RHF or ROHF energy depending on the spin).
            sqmol.mo_energies (list of float): Molecular orbital energies.
            sqmol.mo_occ (list of float): Molecular orbital occupancies (between 0. and 2.).
            sqmol.n_mos (int): Number of molecular orbitals with a given basis set.
            sqmol.n_sos (int): Number of spin-orbitals with a given basis set.

        Add to sqmol:
            self.mo_coeff (ndarray or List[ndarray]): array of molecular orbital coefficients (MO coeffs) if RHF ROHF
                                                        list of arrays [alpha MO coeffs, beta MO coeffs] if UHF

        Args:
            sqmol (SecondQuantizedMolecule): Populated variables of Molecule plus
                sqmol.basis (string): Basis set.
                sqmol.ecp (dict): The effective core potential (ecp) for any atoms in the molecule.
                    e.g. {"C": "crenbl"} use CRENBL ecp for Carbon atoms.
                sqmol.symmetry (bool or str): Whether to use symmetry in RHF or ROHF calculation.
                    Can also specify point group using string. e.g. "Dooh", "D2h", "C2v", ...
                sqmol.uhf (bool): If True, Use UHF instead of RHF or ROHF reference. Default False


        """

        molecule = mol_to_pyscf(sqmol, sqmol.basis, sqmol.symmetry, sqmol.ecp)

        sqmol.mean_field = self.scf.RHF(molecule) if not sqmol.uhf else self.scf.UHF(molecule)
        sqmol.mean_field.verbose = 0

        chkfile_found = False
        if self.chkfile:
            chkfile_found = os.path.exists(self.chkfile)
            sqmol.mean_field.chkfile = self.chkfile

        # Force broken symmetry for uhf calculation when spin is 0 as shown in
        # https://github.com/sunqm/pyscf/blob/master/examples/scf/32-break_spin_symm.py
        if sqmol.uhf and sqmol.spin == 0 and not chkfile_found:
            dm_alpha, dm_beta = sqmol.mean_field.get_init_guess()
            dm_beta[:1, :] = 0
            dm = (dm_alpha, dm_beta)
            sqmol.mean_field.kernel(dm)
        else:
            sqmol.mean_field.init_guess = "chkfile" if chkfile_found else "minao"
            sqmol.mean_field.kernel()

        sqmol.mean_field.analyze()
        if not sqmol.mean_field.converged:
            raise ValueError("Hartree-Fock calculation did not converge")

        if sqmol.symmetry:
            sqmol.mo_symm_ids = list(self.symm.label_orb_symm(sqmol.mean_field.mol, sqmol.mean_field.mol.irrep_id,
                                                              sqmol.mean_field.mol.symm_orb, sqmol.mean_field.mo_coeff))
            irrep_map = {i: s for s, i in zip(molecule.irrep_name, molecule.irrep_id)}
            sqmol.mo_symm_labels = [irrep_map[i] for i in sqmol.mo_symm_ids]
        else:
            sqmol.mo_symm_ids = None
            sqmol.mo_symm_labels = None

        sqmol.mf_energy = sqmol.mean_field.e_tot
        sqmol.mo_energies = sqmol.mean_field.mo_energy
        sqmol.mo_occ = sqmol.mean_field.mo_occ

        sqmol.n_mos = molecule.nao_nr()
        sqmol.n_sos = 2*sqmol.n_mos

        self.mo_coeff = sqmol.mean_field.mo_coeff

    def get_integrals(self, sqmol, mo_coeff=None):
        r"""Computes core constant, one_body, and two-body integrals for all orbitals

        one-body integrals should be in the form
        h[p,q]= \int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx

        two-body integrals should be in the form
        h[p,q,r,s] = \int \phi_p(x) * \phi_q(y) * V_{elec-elec} \phi_r(y) \phi_s(x) dxdy

        Using molecular orbitals \phi_j(x) = \sum_{ij} A_i(x) mo_coeff_{i,j} where A_i(x) are the atomic orbitals.

        For UHF (if sqmol.uhf is True)
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        where one_body and two_body are appropriately sized arrays for each spin sector.

        Args:
            sqmol (SecondQuantizedMolecule) : SecondQuantizedMolecule populated with all variables defined above
            mo_coeff : Molecular orbital coefficients to use for calculating the integrals, instead of self.mo_coeff

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """

        # Pyscf molecule to get integrals.
        pyscf_mol = mol_to_pyscf(sqmol, sqmol.basis, sqmol.symmetry, sqmol.ecp)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if sqmol.uhf:
            one_body, two_body = self.compute_uhf_integrals(sqmol, mo_coeff)
            return float(pyscf_mol.energy_nuc()), one_body, two_body

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(pyscf_mol.energy_nuc())

        # get_hcore is equivalent to int1e_kin + int1e_nuc.
        one_electron_integrals = mo_coeff.T @ sqmol.mean_field.get_hcore() @ mo_coeff

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

    def compute_uhf_integrals(self, sqmol, mo_coeff):
        """Compute 1-electron and 2-electron integrals
        The return is formatted as
        [numpy.ndarray]*2 numpy array h_{pq} for alpha and beta blocks
        [numpy.ndarray]*3 numpy array storing h_{pqrs} for alpha-alpha, alpha-beta, beta-beta blocks

        Args:
            sqmol (SecondQuantizedMolecule): The SecondQuantizedMolecule object to calculated UHF integrals for.
            mo_coeff (List[array]): The molecular orbital coefficients for both spins [alpha, beta]

        Returns:
            List[array], List[array]: One and two body integrals
        """
        # step 1 : find nao, nmo (atomic orbitals & molecular orbitals)

        # molecular orbitals (alpha and beta will be the same)
        # Lets take alpha blocks to find the shape and things

        # molecular orbitals
        nmo = mo_coeff[0].shape[1]
        # atomic orbitals
        nao = mo_coeff[0].shape[0]

        # step 2 : obtain Hcore Hamiltonian in atomic orbitals basis
        hcore = sqmol.mean_field.get_hcore()

        # step 3 : obatin two-electron integral in atomic basis
        eri = self.ao2mo.restore(8, sqmol.mean_field._eri, nao)

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

        # convert this into the physicist ordering for OpenFermion
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Gpqrs has alpha, alphaBeta, Beta blocks
        Gpqrs = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        return hpq, Gpqrs
