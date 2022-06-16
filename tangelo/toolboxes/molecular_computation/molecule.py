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

"""Module containing datastructures for interfacing with this package
functionalities.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
from pyscf import gto, scf, ao2mo, symm, lib
import openfermion
import openfermion.ops.representations as reps
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations.interaction_operator import get_active_space_integrals as of_get_active_space_integrals

from tangelo.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core
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

    # Defined in __post_init__.
    n_atoms: int = field(init=False)
    n_electrons: int = field(init=False)
    n_min_orbitals: int = field(init=False)

    def __post_init__(self):
        mol = self.to_pyscf()
        self.n_atoms = mol.natm
        self.n_electrons = mol.nelectron
        self.n_min_orbitals = mol.nao_nr()

    @property
    def elements(self):
        return [self.xyz[i][0] for i in range(self.n_atoms)]

    @property
    def coords(self):
        return np.array([self.xyz[i][1] for i in range(self.n_atoms)])

    def to_pyscf(self, basis="CRENBL", symmetry=False, ecp=None):
        """Method to return a pyscf.gto.Mole object.

        Args:
            basis (string): Basis set.
            symmetry (bool): Flag to turn symmetry on
            ecp (dict): Dictionary with ecp definition for each atom e.g. {"Cu": "crenbl"}

        Returns:
            pyscf.gto.Mole: PySCF compatible object.
        """

        mol = gto.Mole(atom=self.xyz)
        mol.basis = basis
        mol.charge = self.q
        mol.spin = self.spin
        mol.symmetry = symmetry
        mol.ecp = ecp if ecp else dict()
        mol.build()

        self.xyz = list()
        for sym, xyz in mol._atom:
            self.xyz += [tuple([sym, tuple([x*lib.parameters.BOHR for x in xyz])])]

        return mol

    def to_file(self, filename, format=None):
        """Write molecule geometry to filename in specified format

        Args:
            filename (str): The name of the file to output the geometry.
            format (str): The output type of "raw", "xyz", or "zmat". If None, will be inferred by the filename
        """
        mol = self.to_pyscf()
        mol.tofile(filename, format)

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
    frozen_orbitals: list or int = field(default="frozen_core", repr=False)

    # Defined in __post_init__.
    mf_energy: float = field(init=False)
    mo_energies: list = field(init=False)
    mo_occ: list = field(init=False)

    mean_field: scf = field(init=False)

    n_mos: int = field(init=False)
    n_sos: int = field(init=False)

    active_occupied: list = field(init=False)
    frozen_occupied: list = field(init=False)
    active_virtual: list = field(init=False)
    frozen_virtual: list = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._compute_mean_field()
        self.freeze_mos(self.frozen_orbitals)

    @property
    def n_active_electrons(self):
        return int(sum([self.mo_occ[i] for i in self.active_occupied]))

    @property
    def n_active_sos(self):
        return 2*len(self.active_mos)

    @property
    def n_active_mos(self):
        return len(self.active_mos)

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
            return self.frozen_occupied + self.frozen_virtual
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
        return self.active_occupied + self.active_virtual

    def _compute_mean_field(self):
        """Computes the mean-field for the molecule. It supports open-shell
        mean-field calculation through openfermionpyscf. Depending on the
        molecule spin, it does a restricted or a restriction open-shell
        Hartree-Fock calculation.

        It is also used for defining attributes related to the mean-field
        (mf_energy, mo_energies, mo_occ, n_mos and n_sos).
        """

        molecule = self.to_pyscf(self.basis, self.symmetry, self.ecp)

        self.mean_field = scf.RHF(molecule)
        self.mean_field.verbose = 0
        self.mean_field.kernel()

        if self.symmetry:
            self.mo_symm_ids = list(symm.label_orb_symm(self.mean_field.mol, self.mean_field.mol.irrep_id,
                                                        self.mean_field.mol.symm_orb, self.mean_field.mo_coeff))
            irrep_map = {i: s for s, i in zip(molecule.irrep_name, molecule.irrep_id)}
            self.mo_symm_labels = [irrep_map[i] for i in self.mo_symm_ids]
        else:
            self.mo_symm_ids = None
            self.mo_symm_labels = None

        self.mf_energy = self.mean_field.e_tot
        self.mo_energies = self.mean_field.mo_energy
        self.mo_occ = self.mean_field.mo_occ

        self.n_mos = molecule.nao_nr()
        self.n_sos = 2*self.n_mos

    def _get_fermionic_hamiltonian(self, mo_coeff=None):
        """This method returns the fermionic hamiltonian. It written to take
        into account calls for this function is without argument, and attributes
        are parsed into it.

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.

        Returns:
            FermionOperator: Self-explanatory.
        """

        core_constant, one_body_integrals, two_body_integrals = self.get_active_space_integrals(mo_coeff)

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_body_integrals, two_body_integrals)

        molecular_hamiltonian = reps.InteractionOperator(core_constant, one_body_coefficients, 1 / 2 * two_body_coefficients)

        return get_fermion_operator(molecular_hamiltonian)

    def _convert_frozen_orbitals(self, frozen_orbitals):
        """This method converts an int or a list of frozen_orbitals into four
        categories:
        - Active and occupied MOs;
        - Active and virtual MOs;
        - Frozen and occupied MOs;
        - Frozen and virtual MOs.
        Each of them are list with MOs indexes (first one is 0). Note that they
        are MOs labelled, not spin-orbitals (MOs * 2) indexes.

        Args:
            frozen_orbitals (int or list of int): Number of MOs or MOs indexes
                to freeze.

        Returns:
            list: Nested list of active occupied, frozen occupied, active
                virtual and frozen virtual orbital indexes.
        """

        if frozen_orbitals == "frozen_core":
            frozen_orbitals = get_frozen_core(self.to_pyscf(self.basis)) if not self.ecp else 0
        elif frozen_orbitals is None:
            frozen_orbitals = 0

        # First case: frozen_orbitals is an int.
        # The first n MOs are frozen.
        if isinstance(frozen_orbitals, int):
            frozen_orbitals = list(range(frozen_orbitals))
        # Second case: frozen_orbitals is a list of int.
        # All MOs with indexes in this list are frozen (first MO is 0, second is 1, ...).
        # Everything else raise an exception.
        elif not (isinstance(frozen_orbitals, list) and all(isinstance(_, int) for _ in frozen_orbitals)):
            raise TypeError("frozen_orbitals argument must be an (or a list of) integer(s).")

        occupied = [i for i in range(self.n_mos) if self.mo_occ[i] > 0.]
        virtual = [i for i in range(self.n_mos) if self.mo_occ[i] == 0.]

        frozen_occupied = [i for i in frozen_orbitals if i in occupied]
        frozen_virtual = [i for i in frozen_orbitals if i in virtual]

        # Redefined active orbitals based on frozen ones.
        active_occupied = [i for i in occupied if i not in frozen_occupied]
        active_virtual = [i for i in virtual if i not in frozen_virtual]

        # Calculate number of active electrons and active_mos
        n_active_electrons = round(sum([self.mo_occ[i] for i in active_occupied]))
        n_active_mos = len(active_occupied + active_virtual)

        # Exception raised here if there is no active electron.
        # An exception is raised also if all active orbitals are fully occupied.
        if n_active_electrons == 0:
            raise ValueError("There are no active electrons.")
        if n_active_electrons == 2*n_active_mos:
            raise ValueError("All active orbitals are fully occupied.")

        return active_occupied, frozen_occupied, active_virtual, frozen_virtual

    def freeze_mos(self, frozen_orbitals, inplace=True):
        """This method recomputes frozen orbitals with the provided input."""

        list_of_active_frozen = self._convert_frozen_orbitals(frozen_orbitals)

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
            one_rdm (numpy.array): One-particle density matrix in MO basis.
            two_rdm (numpy.array): Two-particle density matrix in MO basis.

        Returns:
            float: Molecular energy.
        """

        core_constant, one_electron_integrals, two_electron_integrals = self.get_active_space_integrals()

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # reverse back after performing the truncation for frozen orbitals
        two_electron_integrals = two_electron_integrals.transpose(0, 3, 1, 2)

        # Computing the total energy from integrals and provided RDMs.
        e = core_constant + np.sum(one_electron_integrals * one_rdm) + 0.5*np.sum(two_electron_integrals * two_rdm)

        return e.real

    def get_active_space_integrals(self, mo_coeff=None):
        """Computes core constant, one_body, and two-body coefficients with frozen orbitals folded into one-body coefficients
        and core constant

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals

        Returns:
            (float, array, array): (core_constant, one_body coefficients, two_body coefficients)
        """

        return self.get_integrals(mo_coeff, True)

    def get_full_space_integrals(self, mo_coeff=None):
        """Computes core constant, one_body, and two-body integrals for all orbitals

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.

        Returns:
            (float, array, array): (core_constant, one_body coefficients, two_body coefficients)
        """

        return self.get_integrals(mo_coeff, False)

    def get_integrals(self, mo_coeff=None, consider_frozen=True):
        """Computes core constant, one_body, and two-body coefficients for a given active space and mo_coeff

        Args:
            mo_coeff (array): The molecular orbital coefficients to use to generate the integrals.
            consider_frozen (bool): If True, the frozen orbitals are folded into the one_body and core constant terms.

        Returns:
            (float, array, array): (core_constant, one_body coefficients, two_body coefficients)
        """

        # Pyscf molecule to get integrals.
        pyscf_mol = self.to_pyscf(self.basis, self.symmetry, self.ecp)
        if mo_coeff is None:
            mo_coeff = self.mean_field.mo_coeff

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(pyscf_mol.energy_nuc())

        # get_hcore is equivalent to int1e_kin + int1e_nuc.
        one_electron_integrals = mo_coeff.T @ self.mean_field.get_hcore() @ mo_coeff

        # Getting 2-body integrals in atomic and converting to molecular basis.
        two_electron_integrals = ao2mo.kernel(pyscf_mol.intor("int2e"), mo_coeff)
        two_electron_integrals = ao2mo.restore(1, two_electron_integrals, len(mo_coeff))

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # made before performing the truncation for frozen orbitals.
        two_electron_integrals = two_electron_integrals.transpose(0, 2, 3, 1)
        if consider_frozen:
            core_offset, one_electron_integrals, two_electron_integrals = of_get_active_space_integrals(one_electron_integrals,
                                                                                                        two_electron_integrals,
                                                                                                        self.frozen_occupied,
                                                                                                        self.active_mos)

            # Adding frozen electron contribution to core constant.
            core_constant += core_offset

        return core_constant, one_electron_integrals, two_electron_integrals
