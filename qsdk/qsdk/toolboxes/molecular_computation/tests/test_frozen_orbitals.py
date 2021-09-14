import unittest

from qsdk import SecondQuantizedMolecule
from qsdk.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core, get_orbitals_excluding_homo_lumo

H2O = [('O', (0., 0., 0.11779)),
        ('H', (0., 0.75545, -0.47116)),
        ('H', (0., -0.75545, -0.47116))
        ]

mol_h2o = SecondQuantizedMolecule(H2O, 0, 0, "3-21g")


class FrozenOrbitalsTest(unittest.TestCase):

    def test_get_frozen_core(self):
        """ Verify if the frozen orbital suggestions are consistent with
        chemical intuition.
        """

        frozen_h2o = get_frozen_core(mol_h2o)
        self.assertEqual(frozen_h2o, 1)

    def test_get_homo_lumo(self):
        """ Verify if the HOMO-LUMO suggestions are consistent with the provided
        parameters.
        """

        # Getting HOMO-LUMO.
        frozen_homo_lumo = get_orbitals_excluding_homo_lumo(mol_h2o)
        self.assertEqual(frozen_homo_lumo, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12])

        # Active space from HOMO-2 to LUMO+4
        frozen_homo2_lumo4 = get_orbitals_excluding_homo_lumo(mol_h2o, 2, 4)
        self.assertEqual(frozen_homo2_lumo4, [0, 1, 10, 11, 12])


if __name__ == "__main__":
    unittest.main()
