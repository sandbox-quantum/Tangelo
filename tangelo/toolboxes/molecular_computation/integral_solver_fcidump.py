# Copyright SandboxAQ 2021-2024.
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

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver
from tangelo.toolboxes.molecular_computation.integral_solver_pyscf import mol_to_pyscf


class IntegralSolverFCIDUMP(IntegralSolver):
    """Electronic Structure integration when it is already available via a
    FCIDUMP file."""

    def __init__(self, fcidump_file):
        """Initialize the integral solver class for FCIDUMP file.

        Args:
            fcidump_file (string): Path of the FCIDUMP file.
        """
        from pyscf import ao2mo, lib, tools

        self.ao2mo = ao2mo
        self.lib = lib
        self.tools = tools

        if os.path.exists(fcidump_file):
            self.fcidump_file = os.path.abspath(fcidump_file)
        else:
            raise FileNotFoundError(f"The {fcidump_file} file is not found.")

    def set_physical_data(self, mol):
        """Set molecular data that is independant of basis set in mol.

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
        """Retrieves the mean-field corresponding to the FCIDUMP file.

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
        """

        pyscf_mol = mol_to_pyscf(sqmol, sqmol.basis, sqmol.symmetry, sqmol.ecp)

        sqmol.mean_field = self.tools.fcidump.to_scf(self.fcidump_file)
        sqmol.mean_field.verbose = 0

        # Setting max_cycle=0 lets the mean_field updates itself with the
        # current guess, i.e. the one found in the FCIDUMP file.
        sqmol.mean_field.max_cycle = 0
        sqmol.mean_field.kernel()

        sqmol.mf_energy = sqmol.mean_field.e_tot
        sqmol.mo_energies = sqmol.mean_field.mo_energy
        sqmol.mo_occ = sqmol.mean_field.mo_occ

        sqmol.n_mos = pyscf_mol.nao_nr()
        sqmol.n_sos = 2*sqmol.n_mos

        self.mo_coeff = sqmol.mean_field.mo_coeff

    def get_integrals(self, sqmol, mo_coeff=None):
        r"""Retrieves core constant, one_body, and two-body integrals from the
        FCIDUMP file.

        Args:
            sqmol (SecondQuantizedMolecule) : SecondQuantizedMolecule (not used).
            mo_coeff : Molecular orbital coefficients (not used).

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """

        # Reading the FCIDUMP file.
        fcidump_data = self.tools.fcidump.read(self.fcidump_file)

        # Getting the relevant data, like number of orbitals, core constant.
        norb = fcidump_data["NORB"]

        # Reading the nuclear repulsion energy and static coulomb energy,
        # and the electron integrals.
        core_constant = fcidump_data["ECORE"]
        one_electron_integrals = fcidump_data["H1"].reshape((norb,)*2)
        two_electron_integrals = self.ao2mo.restore(1, fcidump_data["H2"], norb)

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # made before performing the truncation for frozen orbitals.
        two_electron_integrals = two_electron_integrals.transpose(0, 2, 3, 1)

        return core_constant, one_electron_integrals, two_electron_integrals
