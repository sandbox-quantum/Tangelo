""" This module defines the MolecularData class, carrying all informations related to a molecular system.
    It also provides related utility functions and methods to perform some classical computations such as
    electronic integrals, etc. """

import openfermion

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


class MolecularData(openfermion.MolecularData):
    """ Currently, this class is coming from openfermion. It will later on be replaced by our own implementation.
        Atom coordinates are assumed to be passed in list format, not string. """

    def __init__(self, mol):
        """ Mol is a pyscf mol or any underlying format we choose to use in qemist to transport molecular data """

        geometry = atom_string_to_list(mol.atom) if isinstance(mol.atom, str) else mol.atom
        self.mol = mol
        openfermion.MolecularData.__init__(self, geometry, mol.basis, mol.spin+1, mol.charge,
                                                        filename="dummy")
        run_pyscf(self, run_scf=True, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
