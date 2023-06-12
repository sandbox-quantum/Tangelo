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

import abc


class IntegralSolver(abc.ABC):
    """Instantiate electronic integral solver"""
    def __init__(self):
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass


class IntegralSolverEmpty(IntegralSolver):
    def __init__(self):
        pass

    def set_physical_data(self, mol):
        pass

    def compute_mean_field(self, sqmol):
        pass

    def get_integrals(self, sqmol, mo_coeff=None):
        pass
