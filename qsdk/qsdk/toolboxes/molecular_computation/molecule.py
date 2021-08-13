"""Docstrings """

from dataclasses import dataclass, field
import numpy as np
from pyscf import gto, scf
from pyscf.gto.mole import Mole
from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData

from qsdk.toolboxes.operators import FermionOperator
from qsdk.electronic_structure_solvers import RHFSolver

# Optional imports (for optional feature)?
import openfermion
import openfermionpyscf


@dataclass
class Molecule:
    xyz: list
    q: int = 0
    spin: int = 0
    n_atoms: int = field(init=False)
    n_electrons: int = field(init=False)
    n_min_orbitals: int = field(init=False)

    def __post_init__(self):
        mol = self.to_pyscf(basis="sto-3g")
        self.n_atoms = len(self.xyz)
        self.n_electrons = mol.nelectron
        self.n_min_orbitals = mol.nao_nr()

    @property
    def elements(self):
        return [self.xyz[i][0] for i in range(self.n_atoms)]

    # Setter for this one?
    @property
    def coords(self):
        return np.array([self.xyz[i][1] for i in range(self.n_atoms)])

    def to_pyscf(self, basis="sto-3g"):
        mol = gto.Mole()
        mol.atom = self.xyz
        mol.basis = basis
        mol.charge = self.q
        mol.spin = self.spin
        mol.build()

        return mol

    # Optional feature.
    def to_openfermion(self, basis="sto-3g"):
        return openfermion.MolecularData(self.xyz, basis, self.spin+1, self.q)


@dataclass
class SecondQuantizedMolecule(Molecule):
    basis: str = "sto-3g"
    frozen_orbitals: list or int = 0
    is_open_shell: bool = False

    mf_energy: float = field(init=False)
    mo_energies: list = field(init=False)
    mo_occ: list = field(init=False)

    n_mos: int = field(init=False)
    n_sos: int = field(init=False)

    active_occupied: list = field(init=False)
    frozen_occupied: list = field(init=False)
    active_virtual: list = field(init=False)
    frozen_virtual: list = field(init=False)

    fermionic_hamiltonian: FermionOperator = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.compute_mean_field()

        # Closed-shell only?
        self.active_occupied = [i for i in range(int(np.ceil(self.n_electrons / 2)))]
        self.active_virtual = [i for i in range(self.n_mos) if i not in self.active_occupied]
        self.frozen_occupied = list()
        self.frozen_virtual = list()

        self._convert_frozen_orbitals(self.frozen_orbitals)
        self.fermionic_hamiltonian = self.get_fermionic_hamiltonian()

    @property
    def n_active_electrons(self):
        return None

    def compute_mean_field(self):
        if self.is_open_shell == False:
            molecule = Molecule(self.xyz, self.q, self.spin)
            solver = RHFSolver(molecule, self.basis)

            self.mf_energy = solver.simulate()
            self.mo_energies = solver.mean_field.mo_energy
            self.mo_occ = solver.mean_field.mo_occ

            self.n_mos = solver.mean_field.mol.nao_nr()
            self.n_sos = 2*self.n_mos
        else:
            raise NotImplementedError

    def get_fermionic_hamiltonian(self):
        """ This method returns the fermionic hamiltonian. It written to take into account
            calls for this function is without argument, and attributes are parsed into it.

            Returns:
                molecular_hamiltonian: An instance of the MolecularOperator class.
                    Indexing is spin up are even numbers, spin down are odd ones.
        """

        occupied_indices = self.frozen_occupied
        active_indices = self.get_active_orbitals()

        of_molecule = self.to_openfermion(self.basis)
        of_molecule = openfermionpyscf.run_pyscf(of_molecule, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)

        molecular_hamiltonian = of_molecule.get_molecular_hamiltonian(occupied_indices, active_indices)
        return openfermion.transforms.get_fermion_operator(molecular_hamiltonian)

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

        # Exception raised here if n_occupied <= frozen_orbitals (int), because it means that there is no active electron.
        # An exception is raised also if all occupied orbitals are in the frozen_orbitals (list).
        if (len(self.active_occupied) == 0) or (len(self.active_virtual) == 0):
            raise ValueError("All electrons or virtual orbitals are frozen in the system.")

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


if __name__ == "__main__":
    #coords = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
    coords = [("O", (0., 0., 0.11779)),
              ("H", (0., 0.75545, -0.47116)),
              ("H", (0., -0.75545, -0.47116))
            ]

    a = Molecule(coords, q=0, spin=0)
    print(a)
    b = SecondQuantizedMolecule(coords, q=0, spin=0, basis="sto-3g", frozen_orbitals=1)
    print(b)
