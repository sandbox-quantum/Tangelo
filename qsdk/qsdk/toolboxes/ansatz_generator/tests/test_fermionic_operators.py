import unittest
from pyscf import gto

from agnostic_simulator import Simulator
from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit, get_vector, vector_to_circuit
from qsdk.toolboxes.ansatz_generator.fermionic_operators import number_operator, spinz_operator, spin2_operator

# Initiate simulator
sim = Simulator(target="qulacs")

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


class fermionic_operators_Test(unittest.TestCase):

    def test_number_operator_hf_ref(self):
        """ Verify that the number penalty terms return zero for all mappings given the correct number of
        electrons and mapping"""

        for mol in [mol_h2, mol_h4]:
            molecule = MolecularData(mol)
            for mapping in ['jw', 'bk', 'scbk']:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=molecule.n_qubits,
                                                         n_electrons=molecule.n_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                num_op = number_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                numval = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(numval, molecule.n_electrons, delta=1e-6)

    def test_spinz_operator_hf_ref(self):
        """ Verify that the number penalty terms return zero for all mappings given the correct number of
        electrons and mapping"""

        for mol in [mol_h2, mol_h4]:
            molecule = MolecularData(mol)
            for mapping in ['jw', 'bk', 'scbk']:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=molecule.n_qubits,
                                                         n_electrons=molecule.n_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                spinz_op = spinz_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                spinzval = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(spinzval, 0.0, delta=1e-6)

    def test_spin2_penalty_terms_hf_ref(self):
        """ Verify that the number penalty terms return zero for all mappings given the correct number of
        S^2 and mapping"""

        for mol in [mol_h2, mol_h4]:
            molecule = MolecularData(mol)
            for mapping in ['jw', 'bk', 'scbk']:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=molecule.n_qubits,
                                                         n_electrons=molecule.n_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                pen_op = spin2_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=pen_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                spin2val = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(spin2val, 0.0, delta=1e-6)

    def test_doublet(self):
        """ Verify that the number penalty terms return zero for all mappings given the correct number of
        S^2 and mapping"""

        for mol in [mol_h2, mol_h4]:
            for mapping in ['jw']:  # may add tests for other mappings later
                molecule = MolecularData(mol)
                vector = get_vector(n_spinorbitals=molecule.n_qubits,
                                    n_electrons=molecule.n_electrons,
                                    mapping=mapping,
                                    up_then_down=True)

                # Add electron with spin up
                vector[molecule.n_electrons//2] = 1

                # Create doublet state prep circuit
                doublet_state_circuit = vector_to_circuit(vector)

                # Test number of electrons is molecule.n_electrons+1
                num_op = number_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty is zero
                numval = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(numval, molecule.n_electrons + 1, delta=1e-6)

                # Test that spin of 0.5 returns zero
                spinz_op = spinz_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty value returned is zero
                spinzval = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(spinzval, 0.5, delta=1e-6)

                # Test s2 penalty for s2=(0.5)*(0.5+1)=0.75 is zero
                spin2_op = spin2_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spin2_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty returned is zero
                spin2val = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(spin2val, 0.5 * (0.5 + 1), delta=1e-6)

    def test_triplet(self):
        """ Verify that the number penalty terms return zero for all mappings given the correct number of
        S^2 and mapping"""

        for mol in [mol_h2, mol_h4]:
            for mapping in ['jw']:
                molecule = MolecularData(mol)
                vector = get_vector(n_spinorbitals=molecule.n_qubits,
                                    n_electrons=molecule.n_electrons,
                                    mapping=mapping,
                                    up_then_down=True)

                # Add electron with spin up and remove electron with spin down
                vector[molecule.n_electrons//2] = 1
                vector[molecule.n_qubits//2 + molecule.n_electrons//2-1] = 0

                # Create triplet state circuit
                triplet_state_circuit = vector_to_circuit(vector)

                # Test number operator
                num_op = number_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty is zero
                numval = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(numval, molecule.n_electrons, delta=1e-6)

                # Test sz value, should be 1 for triplet state
                spinz_op = spinz_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty term is zero
                spinzval = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(spinzval, 1, delta=1e-6)

                # Test S^2 penalty, S2 value shoule be (1)*(1+1)=2 for a triplet state
                spin2_op = spin2_operator(molecule.n_orbitals, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spin2_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=molecule.n_qubits,
                                                             n_electrons=molecule.n_electrons,
                                                             up_then_down=True)

                # Assert penalty returns zero
                spin2val = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(spin2val, 1*(1 + 1), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
