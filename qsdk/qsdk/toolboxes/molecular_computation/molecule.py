"""Docstrings """

from dataclasses import dataclass, field
import numpy as np
from pyscf import gto
import openfermion

@dataclass
class Molecule:
    xyz: list
    q: int = 0
    spin: int = 0
    n_electrons: int = field(init=False)
    n_min_orbitals: int = field(init=False)

    def __post_init__(self):
        mol = self.to_pyscf(basis="sto-3g")
        self.n_electrons = mol.nelectron
        self.n_min_orbitals = mol.nao_nr()

    @property
    def n_atoms(self):
        return len(self.xyz)

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

    def to_openfermion(self, basis="sto-3g"):
        return openfermion.MolecularData(self.xyz, basis, self.spin+1, self.q)

@dataclass
class SecondQuantizedMolecule(Molecule):
    mf: int = 0


if __name__ == "__main__":
    #coords = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
    coords = [("O", (0., 0., 0.11779)),
              ("H", (0., 0.75545, -0.47116)),
              ("H", (0., -0.75545, -0.47116))
            ]

    #mol = Molecule(coords, 0, 0)
    #print(mol.to_openfermion())
    #b = SecondQuantizedMolecule(
