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

import numpy as np

from tangelo.linq import get_backend, Gate, Circuit
from tangelo.toolboxes.optimizers.rotosolve import rotosolve
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator import VariationalCircuitAnsatz


class OptimizerTest(unittest.TestCase):

    def test_rotosolve(self):
        """Test rotosovle on H2 without VQE, using custom variational circuit
        and qubit Hamiltonian with JW qubit mapping on an exact simulator.
        """
        sim = get_backend()
        # Create qubit Hamiltonian compatible with UCC1 Ansatz
        qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=mol_H2_sto3g.fermionic_hamiltonian,
                                                     mapping="jw",
                                                     n_spinorbitals=mol_H2_sto3g.n_active_sos,
                                                     up_then_down=True,)

        # Manual input of UCC1 circuit with extra variational parameters
        circuit = Circuit()
        # Create excitation ladder circuit used to build entangler
        excit_gates = [Gate("RX", 0, parameter=np.pi/2, is_variational=True)]
        excit_gates += [Gate("H", i) for i in {1, 2, 3}]
        excit_gates += [Gate("CNOT", i+1, i) for i in range(3)]
        excit_circuit = Circuit(excit_gates)
        # Build UCC1 circuit: mean field + entangler circuits
        circuit = Circuit([Gate("X", i) for i in {0, 2}])
        circuit += excit_circuit
        circuit.add_gate(Gate("RZ", 3, parameter=0, is_variational=True))
        circuit += excit_circuit.inverse()
        # Translate circuit into variational ansatz
        ansatz = VariationalCircuitAnsatz(circuit)

        # Define function to calculate energy and update variational parameters
        def exp(var_params, ansatz, qubit_hamiltonian):
            ansatz.update_var_params(var_params)
            energy = sim.get_expectation_value(qubit_hamiltonian, ansatz.circuit)
            return energy

        # Run rotosolve, returning energy
        energy, _ = rotosolve(exp, ansatz.var_params_default, ansatz, qubit_hamiltonian)

        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
