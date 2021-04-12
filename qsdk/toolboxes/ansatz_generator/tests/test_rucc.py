import unittest
from pyscf import gto
from openfermion.transforms import get_fermion_operator, jordan_wigner, reorder
from openfermion.utils import up_then_down

from agnostic_simulator import Simulator

from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.ansatz_generator.rucc import RUCC

# Build molecule objects used by the tests.
NaH = [('Na', (0., 0., 0.)), ('H', (0., 0., 1.91439))]
occupied_indices = list(range(5))
active_indices = [5, 9]

mol_nah = gto.Mole()
mol_nah.atom = NaH
mol_nah.basis = "sto-3g"
mol_nah.spin = 0
mol_nah.build()


class UCCSDTest(unittest.TestCase):

    def test_ucc1_NaH(self):
        """ Verify UCC1 functionalities for NaH. """

        molecule = MolecularData(mol_nah)

        # Build circuit
        ucc1_ansatz = RUCC(n_var_params=1)
        ucc1_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices
        )

        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        fermion_hamiltonian = reorder(fermion_hamiltonian,up_then_down)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        ucc1_ansatz.update_var_params([1.002953E-01])
        energy = sim.get_expectation_value(qubit_hamiltonian, ucc1_ansatz.circuit)
        self.assertAlmostEqual(energy, -160.30334364630338, delta=1e-6)

    def test_ucc3_NaH(self):
        """ Verify UCC3 functionalities for NaH. """

        molecule = MolecularData(mol_nah)

        # Build circuit
        ucc3_ansatz = RUCC(n_var_params=3)
        ucc3_ansatz.build_circuit()

        # Build qubit hamiltonian for energy evaluation
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices
        )

        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        fermion_hamiltonian = reorder(fermion_hamiltonian, up_then_down)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

        # Assert energy returned is as expected for given parameters
        sim = Simulator(target="qulacs")
        ucc3_ansatz.update_var_params([-1.876306E-02, -1.687847E-02, 1.030982E-01])
        energy = sim.get_expectation_value(qubit_hamiltonian, ucc3_ansatz.circuit)
        self.assertAlmostEqual(energy, -160.3034595375667, delta=1e-6)

    def test_rucc_wrong_n_params(self):
        """ Verify RUCC wrong number of parameters. """

        with self.assertRaises(ValueError):
            RUCC(n_var_params=999)
        
        with self.assertRaises(ValueError):
            RUCC(n_var_params="3")
        
        with self.assertRaises(ValueError):
            RUCC(n_var_params=3.141516)

        with self.assertRaises(AssertionError):
            ucc3 = RUCC(n_var_params=3)
            ucc3.build_circuit()
            ucc3.update_var_params([3.1415])


if __name__ == "__main__":
    unittest.main()
