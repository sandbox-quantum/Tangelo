import unittest
import numpy as np

from qsdk.problem_decomposition.oniom.oniom import oniom_model
from qsdk.problem_decomposition.oniom.utility import build_molecule, load_text, fragment


class ONIOMTest(unittest.TestCase):

    def test_vqe_cc(self):
        atoms = [('H',(0,0,0)), ('H',(0,0,0.75)),
                ('H',(0,0,2)), ('H',(0,0,2.75))]

        high_vqe = {'method': 'vqe','basis': 'sto-3g'}
        high_cc = {'method': 'ccsd','basis': 'sto-3g'}
        low = {'method': 'rhf','basis': 'sto-3g'}

        units = 'Angstrom'

        system = fragment(solver=low, spin=0, charge=0, units=units)
        model_vqe_1 = fragment(solver=[high_vqe, low], select_atoms=[0, 1], links=None, spin=0, charge=0, units=units)
        model_vqe_2 = fragment(solver=[high_vqe, low], select_atoms=[2, 3], links=None, spin=0, charge=0, units=units)
        model_cc_1 = fragment(solver=[high_cc, low], select_atoms=[0, 1], links=None,spin=0, charge=0, units=units)
        model_cc_2 = fragment(solver=[high_cc, low], select_atoms=[2, 3], links=None,spin=0, charge=0, units=units)

        oniom_model_vqe = oniom_model(atoms, [system,model_vqe_1, model_vqe_2])
        e_tot_vqe = oniom_model_vqe.run()

        oniom_model_cc = oniom_model(atoms, [system, model_cc_1, model_cc_2])
        e_tot_cc = oniom_model_cc.run()
        self.assertAlmostEqual(e_tot_vqe, e_tot_cc, places=5)


if __name__ == "__main__":

    unittest.main()