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

"""Module containing datastructures for interfacing with this package
functionalities.
"""

import copy
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import openfermion
import openfermion.ops.representations as reps
from openfermion.utils import down_index, up_index
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations.interaction_operator import get_active_space_integrals as of_get_active_space_integrals

from tangelo.helpers.utils import is_package_installed
from tangelo.toolboxes.molecular_computation import IntegralSolver, IntegralSolverPsi4, IntegralSolverEmpty
from tangelo.toolboxes.molecular_computation.integral_solver_pyscf import mol_to_pyscf, IntegralSolverPySCF
from tangelo.toolboxes.molecular_computation.frozen_orbitals import convert_frozen_orbitals
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_fermion_operator


def atom_string_to_list(atom_string):
    """Convert atom coordinate string (typically stored in text files) into a
    list/tuple representation suitable for openfermion.MolecularData.
    """

    geometry = []
    for line in atom_string.split("\n"):
        data = line.split()
        if len(data) == 4:
            atom = data[0]
            coordinates = (float(data[1]), float(data[2]), float(data[3]))
            geometry += [(atom, coordinates)]
    return geometry


def molecule_to_secondquantizedmolecule(mol, basis_set="sto-3g", frozen_orbitals=None):
    """Function to convert a Molecule into a SecondQuantizedMolecule.

    Args:
        mol (Molecule): Self-explanatory.
        basis_set (string): String representing the basis set.
        frozen_orbitals (int or list of int): Number of MOs or MOs indexes to
            freeze.

    Returns:
        SecondQuantizedMolecule: Mean-field data structure for a molecule.
    """

    converted_mol = SecondQuantizedMolecule(mol.xyz, mol.q, mol.spin,
                                            basis=basis_set,
                                            frozen_orbitals=frozen_orbitals,
                                            symmetry=mol.symmetry)
    return converted_mol


@dataclass
class Molecule:
    """Custom datastructure to store information about a Molecule. This contains
    only physical information.

    Attributes:
        xyz (array-like or string): Nested array-like structure with elements
            and coordinates (ex:[ ["H", (0., 0., 0.)], ...]). Can also be a
            multi-line string.
        q (float): Total charge.
        spin (int): Absolute difference between alpha and beta electron number.
        solver (IntegralSolver): The class that performs the mean field and integral computation.
        n_atom (int): Self-explanatory.
        n_electrons (int): Self-explanatory.
        n_min_orbitals (int): Number of orbitals with a minimal basis set.

    Properties:
        elements (list): List of all elements in the molecule.
        coords (array of float): N x 3 coordinates matrix.
    """
    xyz: list or str
    q: int = 0
    spin: int = 0
    if is_package_installed("pyscf"):
        default_solver = IntegralSolverPySCF
    elif is_package_installed("psi4"):
        default_solver = IntegralSolverPsi4
    else:
        default_solver = IntegralSolverEmpty

    solver: IntegralSolver = field(default_factory=default_solver)

    # Defined in __post_init__.
    n_atoms: int = field(init=False)
    n_electrons: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.solver, IntegralSolverEmpty):
            raise ValueError("PySCF or Psi4 must be installed or a custom solver (IntegralSolver) instance must be provided.")
        self.solver.set_physical_data(self)

    @property
    def elements(self):
        """(list): List of all elements in the molecule."""
        return [self.xyz[i][0] for i in range(self.n_atoms)]

    @property
    def coords(self):
        """(array of float): N x 3 coordinates matrix."""
        return np.array([self.xyz[i][1] for i in range(self.n_atoms)])

    def to_file(self, filename, format=None):
        """Write molecule geometry to filename in specified format

        Args:
            filename (str): The name of the file to output the geometry.
            format (str): The output type of "raw", "xyz", or "zmat". If None, will be inferred by the filename
                Unless using IntegralSolverPySCF, only "xyz" is supported.
        """
        if isinstance(self.solver, IntegralSolverPySCF):
            mol = mol_to_pyscf(self)
            mol.tofile(filename, format)
        elif filename[-3:] == 'xyz' or format == 'xyz':
            f = open(filename, "w") if format is None else open(filename+".xyz", "w")
            f.write(f"{self.n_atoms}\n")
            f.write("XYZ from Tangelo\n")
            for name, positions in self.xyz:
                f.write(f"{name} {positions[0]} {positions[1]} {positions[2]}\n")
            f.close
        else:
            raise ValueError("Tangelo only supports xyz format unless using IntegralSolverPySCF")

    def to_openfermion(self, basis="sto-3g"):
        """Method to return a openfermion.MolecularData object.

        Args:
            basis (string): Basis set.

        Returns:
            openfermion.MolecularData: Openfermion compatible object.
        """

        return openfermion.MolecularData(self.xyz, basis, self.spin+1, self.q)


@dataclass
class SecondQuantizedMolecule(Molecule):
    """Custom datastructure to store information about a mean field derived
    from a molecule. This class inherits from Molecule and add a number of
    attributes defined by the second quantization.

    Attributes:
        basis (string): Basis set.
        ecp (dict): The effective core potential (ecp) for any atoms in the molecule.
            e.g. {"C": "crenbl"} use CRENBL ecp for Carbon atoms.
        symmetry (bool or str): Whether to use symmetry in RHF or ROHF calculation.
            Can also specify point group using pyscf allowed string.
            e.g. "Dooh", "D2h", "C2v", ...
        uhf (bool): If True, Use UHF instead of RHF or ROHF reference. Default False
        mf_energy (float): Mean-field energy (RHF or ROHF energy depending
            on the spin).
        mo_energies (list of float): Molecular orbital energies.
        mo_occ (list of float): Molecular orbital occupancies (between 0.
            and 2.).
        mean_field (pyscf.scf): Mean-field object (used by CCSD and FCI).
        n_mos (int): Number of molecular orbitals with a given basis set.
        n_sos (int): Number of spin-orbitals with a given basis set.
        active_occupied (list of int): Occupied molecular orbital indexes
            that are considered active.
        frozen_occupied (list of int): Occupied molecular orbital indexes
            that are considered frozen.
        active_virtual (list of int): Virtual molecular orbital indexes
            that are considered active.
        frozen_virtual (list of int): Virtual molecular orbital indexes
            that are considered frozen.
        fermionic_hamiltonian (FermionOperator): Self-explanatory.

    Methods:
        freeze_mos: Change frozen orbitals attributes. It can be done inplace
            or not.

    Properties:
        n_active_electrons (int): Difference between number of total
            electrons and number of electrons in frozen orbitals.
        n_active_sos (int): Number of active spin-orbitals.
        n_active_mos (int): Number of active molecular orbitals.
        frozen_mos (list or None): Frozen MOs indexes.
        actives_mos (list): Active MOs indexes.
    """
    basis: str = "sto-3g"
    ecp: dict = field(default_factory=dict)
    symmetry: bool = False
    uhf: bool = False
    frozen_orbitals: list or int = field(default="frozen_core", repr=False)

    # Defined in __post_init__.
    mf_energy: float = field(init=False)
    mo_energies: list = field(init=False)
    mo_occ: list = field(init=False)

    n_mos: int = field(init=False)
    n_sos: int = field(init=False)

    active_occupied: list = field(init=False)
    frozen_occupied: list = field(init=False)
    active_virtual: list = field(init=False)
    frozen_virtual: list = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.solver.compute_mean_field(self)
        self.freeze_mos(self.frozen_orbitals)

    @property
    def n_active_electrons(self):
        return sum(self.n_active_ab_electrons)

    @property
    def n_active_ab_electrons(self):
        if self.uhf:
            return (int(sum([self.mo_occ[0][i] for i in self.active_occupied[0]])), int(sum([self.mo_occ[1][i] for i in self.active_occupied[1]])))
        else:
            n_active_electrons = int(sum([self.mo_occ[i] for i in self.active_occupied]))
            n_alpha = n_active_electrons//2 + self.spin//2 + (n_active_electrons % 2)
            n_beta = n_active_electrons//2 - self.spin//2
            return (n_alpha, n_beta)

    @property
    def n_active_sos(self):
        return 2*len(self.active_mos) if not self.uhf else max(len(self.active_mos[0])*2, len(self.active_mos[1])*2)

    @property
    def n_active_mos(self):
        return len(self.active_mos) if not self.uhf else [len(self.active_mos[0]), len(self.active_mos[1])]

    @property
    def fermionic_hamiltonian(self):
        return self._get_fermionic_hamiltonian()

    @property
    def frozen_mos(self):
        """This property returns MOs indexes for the frozen orbitals. It was
        written to take into account if one of the two possibilities (occ or
        virt) is None. In fact, list + None, None + list or None + None return
        an error. An empty list cannot be sent because PySCF mp2 returns
        "IndexError: list index out of range".

        Returns:
            list: MOs indexes frozen (occupied + virtual).
        """
        if self.frozen_occupied and self.frozen_virtual:
            return (self.frozen_occupied + self.frozen_virtual if not self.uhf else
                    [self.frozen_occupied[0] + self.frozen_virtual[0], self.frozen_occupied[1] + self.frozen_virtual[1]])
        elif self.frozen_occupied:
            return self.frozen_occupied
        elif self.frozen_virtual:
            return self.frozen_virtual
        else:
            return None

    @property
    def active_mos(self):
        """This property returns MOs indexes for the active orbitals.

        Returns:
            list: MOs indexes that are active (occupied + virtual).
        """
        return self.active_occupied + self.active_virtual if not self.uhf else [self.active_occupied[i]+self.active_virtual[i] for i in range(2)]

    @property
    def active_spin(self):
        """This property returns the spin of the active space.

        Returns:
            int: n_alpha - n_beta electrons of the active occupied orbital space.
        """
        n_alpha, n_beta = self.n_active_ab_electrons
        return n_alpha - n_beta

    @property
    def mo_coeff(self):
        """This property returns the molecular orbital coefficients.

        Returns:
            np.array: MO coefficient as a numpy array.
        """
        return self.solver.mo_coeff

    @mo_coeff.setter
    def mo_coeff(self, new_mo_coeff):
        # Asserting the new molecular coefficient matrix have the same dimensions.
        assert self.solver.mo_coeff.shape == (new_mo_coeff := np.array(new_mo_coeff)).shape, \
            f"The new molecular coefficients matrix has a {new_mo_coeff.shape}"\
            f" shape: expected shape is {self.solver.mo_coeff.shape}."
        self.solver.mo_coeff = new_mo_coeff

    def _get_fermionic_hamiltonian(self, mo_coeff=None):
        """This method returns the fermionic hamiltonian. It written to take
        into account calls for this function is without argument, and attributes
        are parsed into it.

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.

        Returns:
            FermionOperator: Self-explanatory.
        """

        if self.uhf:
            return get_fermion_operator(self._get_molecular_hamiltonian_uhf())
        core_constant, one_body_integrals, two_body_integrals = self.get_active_space_integrals(mo_coeff)

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_body_integrals, two_body_integrals)

        molecular_hamiltonian = reps.InteractionOperator(core_constant, one_body_coefficients, 1 / 2 * two_body_coefficients)

        return get_fermion_operator(molecular_hamiltonian)

    def freeze_mos(self, frozen_orbitals, inplace=True):
        """This method recomputes frozen orbitals with the provided input."""

        list_of_active_frozen = convert_frozen_orbitals(self, frozen_orbitals)

        if not self.uhf:
            if any([self.mo_occ[i] == 1 for i in list_of_active_frozen[1]]):
                raise NotImplementedError("Freezing half-filled orbitals is not implemented yet for RHF/ROHF.")

        if inplace:
            self.frozen_orbitals = frozen_orbitals

            self.active_occupied = list_of_active_frozen[0]
            self.frozen_occupied = list_of_active_frozen[1]
            self.active_virtual = list_of_active_frozen[2]
            self.frozen_virtual = list_of_active_frozen[3]

            return None
        else:
            # Shallow copy is enough to copy the same object and changing frozen
            # orbitals (deepcopy also returns errors).
            copy_self = copy.copy(self)

            copy_self.frozen_orbitals = frozen_orbitals

            copy_self.active_occupied = list_of_active_frozen[0]
            copy_self.frozen_occupied = list_of_active_frozen[1]
            copy_self.active_virtual = list_of_active_frozen[2]
            copy_self.frozen_virtual = list_of_active_frozen[3]

            return copy_self

    def energy_from_rdms(self, one_rdm, two_rdm):
        """Computes the molecular energy from one- and two-particle reduced
        density matrices (RDMs). Coefficients (integrals) are computed
        on-the-fly using a pyscf object and the mean-field. Frozen orbitals
        are supported with this method.

        Args:
            one_rdm (array or List[array]): One-particle density matrix in MO basis.
                If UHF [alpha one_rdm, beta one_rdm]
            two_rdm (array or List[array]): Two-particle density matrix in MO basis.
                If UHF [alpha-alpha two_rdm, alpha-beta two_rdm, beta-beta two_rdm]

        Returns:
            float: Molecular energy.
        """

        core_constant, one_electron_integrals, two_electron_integrals = self.get_active_space_integrals()

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # reverse back after performing the truncation for frozen orbitals
        if self.uhf:
            two_electron_integrals = [two_electron_integrals[i].transpose(0, 3, 1, 2) for i in range(3)]
            factor = [1/2, 1, 1/2]
            e = (core_constant +
                 np.sum([np.sum(one_electron_integrals[i] * one_rdm[i]) for i in range(2)]) +
                 np.sum([np.sum(two_electron_integrals[i] * two_rdm[i]) * factor[i] for i in range(3)]))
        else:
            two_electron_integrals = two_electron_integrals.transpose(0, 3, 1, 2)

            # Computing the total energy from integrals and provided RDMs.
            e = core_constant + np.sum(one_electron_integrals * one_rdm) + 0.5*np.sum(two_electron_integrals * two_rdm)

        return e.real

    def get_integrals(self, mo_coeff=None, fold_frozen=True):
        """Computes core constant, one_body, and two-body coefficients with frozen orbitals folded into one-body coefficients
        and core constant for mo_coeff if fold_frozen is True

        For UHF
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals
            fold_frozen (bool): If False, the full integral matrices and core constant are returned.
                If True, the core constant, one_body, and two-body coefficients are calculated with frozen orbitals
                folded into one-body coefficients and core constant. Default True

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """
        if not self.uhf:
            core_constant, one_body_integrals, two_body_integrals = self.solver.get_integrals(self, mo_coeff)
            if fold_frozen:
                core_offset, one_body_integrals, two_body_integrals = of_get_active_space_integrals(one_body_integrals,
                                                                                                    two_body_integrals,
                                                                                                    self.frozen_occupied,
                                                                                                    self.active_mos)

                # Adding frozen electron contribution to core constant.
                core_constant += core_offset
        else:
            core_constant, one_body_integrals, two_body_integrals = self.solver.get_integrals(self, mo_coeff)
            if fold_frozen:
                core_constant, one_body_integrals, two_body_integrals = self._get_active_space_integrals_uhf(core_constant,
                                                                                                             one_body_integrals,
                                                                                                             two_body_integrals)
        return core_constant, one_body_integrals, two_body_integrals

    def get_active_space_integrals(self, mo_coeff=None):
        """Computes core constant, one_body, and two-body coefficients with frozen orbitals folded into one-body coefficients
        and core constant
        For UHF
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """

        return self.get_integrals(mo_coeff, True)

    def get_full_space_integrals(self, mo_coeff=None):
        """Computes core constant, one_body, and two-body integrals for all orbitals
        For UHF
        one_body coefficients are [alpha one_body, beta one_body]
        two_body coefficients are [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body]

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.

        Returns:
            (float, array or List[array], array or List[array]): (core_constant, one_body coefficients, two_body coefficients)
        """

        return self.get_integrals(mo_coeff, False)

    def _get_active_space_integrals_uhf(self, core_constant, one_body_integrals, two_body_integrals, occupied_indices=None, active_indices=None):
        """Get active space integrals with uhf reference
        The return is
        (core_constant,
        [alpha one_body, beta one_body],
        [alpha-alpha two_body, alpha-beta two_body, beta-beta two_body])

        Args:
            occupied_indices (array-like): The frozen occupied orbital indices
            active_indices (array-like): The active orbital indices
            mo_coeff (List[array]): The molecular orbital coefficients to use to generate the integrals.

        Returns:
            (float, List[array], List[array]): Core constant, one body integrals, two body integrals
        """

        occupied_indices = self.frozen_occupied if occupied_indices is None else occupied_indices
        active_indices = self.active_mos if active_indices is None else active_indices
        if (len(active_indices) < 1):
            raise ValueError('Some active indices required for reduction.')

        # alpha part
        for i in occupied_indices[0]:
            core_constant += one_body_integrals[0][i, i]
            # alpha part of j
            for j in occupied_indices[0]:
                core_constant += 0.5*(two_body_integrals[0][i, j, j, i]-two_body_integrals[0][i, j, i, j])
            # beta part of j
            for j in occupied_indices[1]:
                core_constant += 0.5*(two_body_integrals[1][i, j, j, i])

        # beta part
        for i in occupied_indices[1]:
            core_constant += one_body_integrals[1][i, i]
            # alpha part of j
            for j in occupied_indices[0]:
                core_constant += 0.5*(two_body_integrals[1][j, i, i, j])   # i, j are swaped to make BetaAlpha same as AlphaBeta
            # beta part of j
            for j in occupied_indices[1]:
                core_constant += 0.5*(two_body_integrals[2][i, j, j, i]-two_body_integrals[2][i, j, i, j])

        # Modified one electron integrals
        one_body_integrals_new_aa = np.copy(one_body_integrals[0])
        one_body_integrals_new_bb = np.copy(one_body_integrals[1])

        # alpha alpha block
        for u, v in product(active_indices[0], repeat=2):         # u is u_a, v i v_a
            for i in occupied_indices[0]:  # i belongs to alpha block
                one_body_integrals_new_aa[u, v] += (two_body_integrals[0][i, u, v, i] - two_body_integrals[0][i, u, i, v])
            for i in occupied_indices[1]:  # i belongs to beta block
                one_body_integrals_new_aa[u, v] += two_body_integrals[1][u, i, i, v]  # I am swaping u,v with I; to make AlphaBeta

        # beta beta block
        for u, v in product(active_indices[1], repeat=2):         # u is u_beta, v i v_beta
            for i in occupied_indices[1]:  # i belongs to beta block
                one_body_integrals_new_bb[u, v] += (two_body_integrals[2][i, u, v, i] - two_body_integrals[2][i, u, i, v])
            for i in occupied_indices[0]:  # i belongs to alpha block
                one_body_integrals_new_bb[u, v] += two_body_integrals[1][i, u, v, i]  # this is AlphaBeta

        one_body_integrals_new = [one_body_integrals_new_aa[np.ix_(active_indices[0], active_indices[0])],
                                  one_body_integrals_new_bb[np.ix_(active_indices[1], active_indices[1])]]

        TwInt_aa = two_body_integrals[0][np.ix_(active_indices[0], active_indices[0],
                                                active_indices[0], active_indices[0])]

        TwInt_bb = two_body_integrals[2][np.ix_(active_indices[1], active_indices[1],
                                                active_indices[1], active_indices[1])]

        # (alpha|BetaBeta|alpha) is the format of openfermion InteractionOperator

        TwInt_ab = two_body_integrals[1][np.ix_(active_indices[0], active_indices[1],
                                                active_indices[1], active_indices[0])]

        two_body_integrals_new = [TwInt_aa, TwInt_ab, TwInt_bb]

        return core_constant, one_body_integrals_new, two_body_integrals_new

    def _get_molecular_hamiltonian_uhf(self):
        """Output arrays of the second quantized Hamiltonian coefficients.
        Note:
            The indexing convention used is that even indices correspond to
            spin-up (alpha) modes and odd indices correspond to spin-down
            (beta) modes.

        Returns:
            InteractionOperator: The molecular hamiltonian
        """

        constant, one_body_integrals, two_body_integrals = self.get_active_space_integrals()

        # Lets find the dimensions
        n_orb_a = one_body_integrals[0].shape[0]
        n_orb_b = one_body_integrals[1].shape[0]

        # TODO: Implement more compact ordering. May be possible by defining own up_index and down_index functions
        # Instead of
        # n_qubits = n_orb_a + n_orb_b
        # We use
        n_qubits = 2*max(n_orb_a, n_orb_b)

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        # aa
        for p, q in product(range(n_orb_a), repeat=2):
            pi = up_index(p)
            qi = up_index(q)
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[pi, qi] = one_body_integrals[0][p, q]
            for r, s in product(range(n_orb_a), repeat=2):
                two_body_coefficients[pi, qi, up_index(r), up_index(s)] = (two_body_integrals[0][p, q, r, s] / 2.)

        # bb
        for p, q in product(range(n_orb_b), repeat=2):
            pi = down_index(p)
            qi = down_index(q)
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[pi, qi] = one_body_integrals[1][p, q]
            for r, s in product(range(n_orb_b), repeat=2):
                two_body_coefficients[pi, qi, down_index(r), down_index(s)] = (two_body_integrals[2][p, q, r, s] / 2.)

        # abba
        for p, q, r, s in product(range(n_orb_a), range(n_orb_b), range(n_orb_b), range(n_orb_a)):
            two_body_coefficients[up_index(p), down_index(q), down_index(r), up_index(s)] = (two_body_integrals[1][p, q, r, s] / 2.)

        # baab
        for p, q, r, s in product(range(n_orb_b), range(n_orb_a), range(n_orb_a), range(n_orb_b)):
            two_body_coefficients[down_index(p), up_index(q), up_index(r), down_index(s)] = (two_body_integrals[1][q, p, s, r] / 2.)

        # Cast to InteractionOperator class and return.
        molecular_hamiltonian = openfermion.InteractionOperator(constant, one_body_coefficients, two_body_coefficients)

        return molecular_hamiltonian
