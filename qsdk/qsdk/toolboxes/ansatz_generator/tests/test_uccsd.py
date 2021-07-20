import unittest
import numpy as np
from pyscf import gto

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.qubit_mappings import jordan_wigner
from qsdk.toolboxes.ansatz_generator.uccsd import UCCSD

from agnostic_simulator import Simulator

# Build molecule objects used by the tests
H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
H4 = [['H', [0.7071067811865476, 0.0, 0.0]], ['H', [0.0, 0.7071067811865476, 0.0]],
      ['H', [-1.0071067811865476, 0.0, 0.0]], ['H', [0.0, -1.0071067811865476, 0.0]]]

mol_h2 = gto.Mole()
mol_h2.atom = H2
mol_h2.basis = "sto-3g"
mol_h2.spin = 0
mol_h2.build()

mol_h4 = gto.Mole()
mol_h4.atom = H4
mol_h4.basis = "sto-3g"
mol_h4.spin = 0
mol_h4.build()

mol_h4_open = gto.Mole()
mol_h4_open.atom = H4
mol_h4_open.basis = "sto-3g"
mol_h4_open.charge = 1
mol_h4_open.spin = 1
mol_h4_open.build()


class UCCSDTest(unittest.TestCase):

    def test_uccsd_set_var_params(self):
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array).
        MP2 have their own tests """

        molecule = MolecularData(mol_h2)
        uccsd_ansatz = UCCSD(molecule)

        uccsd_ansatz.set_var_params("ones")
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, np.array([1., 1.]), decimal=6)

        uccsd_ansatz.set_var_params([1., 1.])
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, np.array([1., 1.]), decimal=6)

        uccsd_ansatz.set_var_params(np.array([1., 1.]))
        np.testing.assert_array_almost_equal(uccsd_ansatz.var_params, np.array([1., 1.]), decimal=6)

    def test_uccsd_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """
        molecule = MolecularData(mol_h2)
        uccsd_ansatz = UCCSD(molecule)

        self.assertRaises(ValueError, uccsd_ansatz.set_var_params, np.array([1., 1., 1., 1.]))

    def test_uccsd_set_params_MP2_H2(self):
        """ Verify closed-shell UCCSD functionalities for H2: MP2 initial parameters """

        molecule = MolecularData(mol_h2)

        uccsd_ansatz = UCCSD(molecule)
        uccsd_ansatz.set_var_params("MP2")

        expected = [2e-05, 0.0363253711023451]
        self.assertAlmostEqual(np.linalg.norm(uccsd_ansatz.var_params), np.linalg.norm(expected), delta=1e-10)

    def test_uccsd_set_params_MP2_H4(self):
        """ Verify closed-shell UCCSD functionalities for H4: MP2 initial parameters """

        molecule = MolecularData(mol_h4)

        uccsd_ansatz = UCCSD(molecule)
        uccsd_ansatz.set_var_params("MP2")

        expected = [2e-05, 2e-05, 2e-05, 2e-05, 0.03894901872789466, 0.07985689676283764, 0.02019977190077326,
                    0.03777151472046017, 0.05845449631119356, 0.017956568628560945, -0.07212522602179856,
                    -0.03958975799697206, -0.042927857009029735, -0.025307140867721886]
        self.assertAlmostEqual(np.linalg.norm(uccsd_ansatz.var_params), np.linalg.norm(expected), delta=1e-10)

    def test_uccsd_H2(self):
        """ Verify closed-shell UCCSD functionalities for H2 """

        molecule = MolecularData(mol_h2)

        # Build circuit
        uccsd_ansatz = UCCSD(molecule)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        uccsd_ansatz.update_var_params([5.86665842e-06, 0.0565317429])
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.137270174551959, delta=1e-6)

    def test_uccsd_H4_open(self):
        """ Verify close-shell UCCSD functionalities for H4 """

        molecule = MolecularData(mol_h4_open)
        var_params = [-3.68699814e-03,  1.96987010e-02, -1.12573056e-03,  1.31279980e-04,
                      -4.78466616e-02, -8.56400635e-05,  1.60914422e-03, -2.92334744e-02,
                       3.08405067e-03,  1.39404070e-01,  2.42040971e-02, -2.80714763e-03,
                       1.34675820e-01, -7.70867505e-02,  9.86158364e-05,  2.54962567e-02,
                       5.78169071e-02,  2.46873743e-03, -1.05736505e-01, -4.22089003e-02]

        # Build circuit
        uccsd_ansatz = UCCSD(molecule)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        uccsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.639461490, delta=1e-6)

    def test_uccsd_H4(self):
        """ Verify closed-shell UCCSD functionalities for H4 """

        molecule = MolecularData(mol_h4)
        var_params = [-3.96898484e-04, 4.59786847e-05, 3.95285013e-05, 1.85885610e-04, 1.05759154e-02,
                      3.47363359e-01, 3.42657596e-02, 1.45006203e-02, 7.43941871e-02, 7.57255601e-03,
                      -1.83407761e-01, -1.03261491e-01, 1.34258277e-02, -3.78096407e-02]

        # Build circuit
        uccsd_ansatz = UCCSD(molecule)
        uccsd_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        uccsd_ansatz.update_var_params(var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, uccsd_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9778041, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
