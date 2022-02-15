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

import unittest

from tangelo.linq import Simulator
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit, get_vector, vector_to_circuit
from tangelo.toolboxes.ansatz_generator.fermionic_operators import number_operator, spinz_operator, spin2_operator

# Initiate simulator
sim = Simulator()


class fermionic_operators_Test(unittest.TestCase):

    def test_number_operator_hf_ref(self):
        """Verify that the number penalty terms return zero for all mappings
        given the correct number of electrons and mapping.
        """

        for mol in [mol_H2_sto3g, mol_H4_sto3g]:
            for mapping in ["jw", "bk", "scbk"]:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=mol.n_active_sos,
                                                         n_electrons=mol.n_active_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                num_op = number_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                numval = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(numval, mol.n_active_electrons, delta=1e-6)

    def test_spinz_operator_hf_ref(self):
        """Verify that the number penalty terms return zero for all mappings
        given the correct number of electrons and mapping.
        """

        for mol in [mol_H2_sto3g, mol_H4_sto3g]:
            for mapping in ["jw", "bk", "scbk"]:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=mol.n_active_sos,
                                                         n_electrons=mol.n_active_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                spinz_op = spinz_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                spinzval = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(spinzval, 0.0, delta=1e-6)

    def test_spin2_penalty_terms_hf_ref(self):
        """Verify that the number penalty terms return zero for all mappings
        given the correct number of S^2 and mapping.
        """

        for mol in [mol_H2_sto3g, mol_H4_sto3g]:
            for mapping in ["jw", "bk", "scbk"]:
                hf_state_circuit = get_reference_circuit(n_spinorbitals=mol.n_active_sos,
                                                         n_electrons=mol.n_active_electrons,
                                                         mapping=mapping,
                                                         up_then_down=True)
                pen_op = spin2_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=pen_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert energy returned is as expected for given parameters
                spin2val = sim.get_expectation_value(qubit_hamiltonian, hf_state_circuit)
                self.assertAlmostEqual(spin2val, 0.0, delta=1e-6)

    def test_doublet(self):
        """Verify that the number penalty terms return zero for all mappings
        given the correct number of S^2 and mapping.
        """

        for mol in [mol_H2_sto3g, mol_H4_sto3g]:
            for mapping in ["jw"]:  # may add tests for other mappings later
                vector = get_vector(n_spinorbitals=mol.n_active_sos,
                                    n_electrons=mol.n_active_electrons,
                                    mapping=mapping,
                                    up_then_down=True)

                # Add electron with spin up
                vector[mol.n_active_electrons//2] = 1

                # Create doublet state prep circuit
                doublet_state_circuit = vector_to_circuit(vector)

                # Test number of electrons is mol.n_active_electrons+1
                num_op = number_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty is zero
                numval = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(numval, mol.n_active_electrons + 1, delta=1e-6)

                # Test that spin of 0.5 returns zero
                spinz_op = spinz_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty value returned is zero
                spinzval = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(spinzval, 0.5, delta=1e-6)

                # Test s2 penalty for s2=(0.5)*(0.5+1)=0.75 is zero
                spin2_op = spin2_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spin2_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty returned is zero
                spin2val = sim.get_expectation_value(qubit_hamiltonian, doublet_state_circuit)
                self.assertAlmostEqual(spin2val, 0.5 * (0.5 + 1), delta=1e-6)

    def test_triplet(self):
        """Verify that the number penalty terms return zero for all mappings
        given the correct number of S^2 and mapping.
        """

        for mol in [mol_H2_sto3g, mol_H4_sto3g]:
            for mapping in ["jw"]:
                vector = get_vector(n_spinorbitals=mol.n_active_sos,
                                    n_electrons=mol.n_active_electrons,
                                    mapping=mapping,
                                    up_then_down=True)

                # Add electron with spin up and remove electron with spin down
                vector[mol.n_active_electrons//2] = 1
                vector[mol.n_active_sos//2 + mol.n_active_electrons//2-1] = 0

                # Create triplet state circuit
                triplet_state_circuit = vector_to_circuit(vector)

                # Test number operator
                num_op = number_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=num_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty is zero
                numval = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(numval, mol.n_active_electrons, delta=1e-6)

                # Test sz value, should be 1 for triplet state
                spinz_op = spinz_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spinz_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty term is zero
                spinzval = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(spinzval, 1, delta=1e-6)

                # Test S^2 penalty, S2 value shoule be (1)*(1+1)=2 for a triplet state
                spin2_op = spin2_operator(mol.n_active_mos, up_then_down=False)
                qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=spin2_op,
                                                             mapping=mapping,
                                                             n_spinorbitals=mol.n_active_sos,
                                                             n_electrons=mol.n_active_electrons,
                                                             up_then_down=True)

                # Assert penalty returns zero
                spin2val = sim.get_expectation_value(qubit_hamiltonian, triplet_state_circuit)
                self.assertAlmostEqual(spin2val, 1*(1 + 1), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
