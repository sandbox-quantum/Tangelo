""" This module defines the MolecularData class, carrying all informations related to a molecular system.
    It also provides related utility functions and methods to perform some classical computations such as
    electronic integrals, etc. """

import numpy as np
import openfermion
from pyscf import ao2mo

from .integral_calculation import run_pyscf

def atom_string_to_list(atom_string):
    """ Convert atom coordinate string (typically stored in text files) into a list/tuple representation
        suitable for MolecularData """

    geometry = []
    for line in atom_string.split("\n"):
        data = line.split()
        if len(data) == 4:
            atom = data[0]
            coordinates = (float(data[1]), float(data[2]), float(data[3]))
            geometry += [(atom, coordinates)]
    return geometry

def suggest_frozen_orbitals(molecule):
    """Function to compute de number of frozen orbitals. This function is only
    for the core (occupied orbitals).

    Args:
        molecule (pyscf.gto): Molecule to be evaluated.
    """

    # Freezing core of each atom.
    core_orbitals = {
        "H": 0, "He": 0,
        "Li": 1, "Be": 1, "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
    }

    frozen_core = 0

    # Copunting how many of each element is in the molecule.
    elements = {i: molecule.elements.count(i) for i in molecule.elements}
    for k, v in elements.items():
        frozen_core += v * core_orbitals[k]

    return frozen_core


class MolecularData(openfermion.MolecularData):
    """ Currently, this class is coming from openfermion. It will later on be replaced by our own implementation.
        Atom coordinates are assumed to be passed in list format, not string.

        Args:
            mol (pyscf.gto): Format we choose to use in qemist to transport molecular data.
            frozen_orbitals (int or list of int): Optional argument to freeze MOs.
    """

    def __init__(self, mol, mean_field=None, frozen_orbitals=None):

        geometry = atom_string_to_list(mol.atom) if isinstance(mol.atom, str) else mol.atom
        self.mol = mol

        openfermion.MolecularData.__init__(self, geometry, mol.basis, mol.spin+1, mol.charge)
        run_pyscf(self, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        # Overwrite Openfermion object with information consistent with mean-field
        if mean_field:
            self.mf = mean_field
            self.n_atoms = mol.natm
            self.atoms = [row[0] for row in mol.atom],
            self.protons = 0
            self.nuclear_repulsion = mol.energy_nuc()
            self.charge = mol.charge
            self.n_electrons = mol.nelectron
            self.n_orbitals = len(mean_field.mo_energy)
            self.n_spin_orbitals = 2 * self.n_orbitals
            self.hf_energy = mean_field.e_tot
            self.orbital_energies = mean_field.mo_energy
            self.mp2_energy = None
            self.cisd_energy = None
            self.fci_energy = None
            self.ccsd_energy = None
            self.general_calculations = {}
            self._canonical_orbitals = mean_field.mo_coeff
            self._overlap_integrals = mean_field.get_ovlp()
            self.h_core = mean_field.get_hcore()
            self._one_body_integrals = self._canonical_orbitals.T @ self.h_core @ self._canonical_orbitals
            twoint = mean_field._eri
            eri = ao2mo.restore(8, twoint, self.n_orbitals)
            eri = ao2mo.incore.full(eri, self._canonical_orbitals)
            eri = ao2mo.restore(1, eri, self.n_orbitals)
            self._two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')
            self.n_qubits = self.n_spin_orbitals

        # By default, all orbitals are active.
        self.active_occupied = [i for i in range(int(np.ceil(self.n_electrons / 2)))]
        self.active_virtual = [i for i in range(self.n_orbitals) if i not in self.active_occupied]
        self.frozen_occupied = list()
        self.frozen_virtual = list()

        # If frozen_orbitals is not None, 0 nor [], the convert function is called.
        # The four previous attributes (active occ, active virt, frozen occ, frozen virt)
        # are expected to change.
        if frozen_orbitals:
            self._convert_frozen_orbitals(frozen_orbitals)

    def _convert_frozen_orbitals(self, frozen_orbitals):
        """ This method converts an int or a list of frozen_orbitals into 4 categories:
                - Active and occupied MOs;
                - Active and virtual MOs;
                - Frozen and occupied MOs;
                - Frozen and virtual MOs.
            Each of them are list with MOs indexes (first one is 0).
            Note that they are MOs, not spin-orbitals (MOs * 2).

            Args:
                frozen_orbitals (int or list of int): Number of MOs or MOs indexes to freeze.
        """

        # First case: frozen_orbitals is an int.
        # The first n MOs are frozen.
        if isinstance(frozen_orbitals, int):
            self.frozen_occupied = [i for i in range(frozen_orbitals) if i < frozen_orbitals]
            self.frozen_virtual = [i for i in range(frozen_orbitals) if i >= frozen_orbitals]
        # Second case: frozen_orbitals is a list of int.
        # All MOs with indexes in this list are frozen (first MO is 0, second is 1, ...).
        elif isinstance(frozen_orbitals, list) and all(isinstance(_, int) for _ in frozen_orbitals):
            n_occupied = int(np.ceil(self.n_electrons / 2))
            self.frozen_occupied = [i for i in frozen_orbitals if i < n_occupied]
            self.frozen_virtual = [i for i in frozen_orbitals if i not in self.frozen_occupied]
        # Everything else raise an exception.
        else:
            raise TypeError("frozen_orbitals argument must be an (or a list of) integer(s).")

        # Redefined active orbitals based on frozen ones.
        self.active_occupied = [i for i in self.active_occupied if i not in self.frozen_occupied]
        self.active_virtual = [i for i in self.active_virtual if i not in self.frozen_virtual]

        # Useful for the making of the ansatz and convertion into qubits and gates.
        # Frozen things are not considered anymore.
        self.n_qubits = len(self.active_occupied + self.active_virtual) * 2
        self.n_orbitals = len(self.active_occupied + self.active_virtual)
        self.n_electrons = len(self.active_occupied) * 2

        # Exception raised here if n_occupied <= frozen_orbitals (int), because it means that there is no active electron.
        # An exception is raised also if all occupied orbitals are in the frozen_orbitals (list).
        if (len(self.active_occupied) == 0) or (len(self.active_virtual) == 0):
            raise ValueError("All electrons or virtual orbitals are frozen in the system.")

    def get_molecular_hamiltonian(self):
        """ This method returns the fermionic hamiltonian. It written to take into account
            calls for this function is without argument, and attributes are parsed into it.

            Returns:
                molecular_hamiltonian: An instance of the MolecularOperator class.
                    Indexing is spin up are even numbers, spin down are odd ones.
        """

        occupied_indices = self.frozen_occupied
        active_indices = self.get_active_orbitals()

        return super().get_molecular_hamiltonian(occupied_indices, active_indices)

    def get_frozen_orbitals(self):
        """ This method returns MOs indexes for the frozen orbitals. It was written
            to take into account if one of the two possibilities (occ or virt) is
            None. In fact, list + None, None + list or None + None return an error.
            An empty list cannot be sent because PySCF mp2 returns "IndexError: list index out of range".

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

    def get_active_orbitals(self):
        """ This method returns MOs indexes for the active orbitals.

            Returns:
                list: MOs indexes that are active (occupied + virtual).
        """
        return self.active_occupied + self.active_virtual
