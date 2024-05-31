# Copyright SandboxAQ 2021-2024.
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
from tangelo.toolboxes.optimizers.rotosolve import rotosolve, rotoselect
from tangelo.toolboxes.operators.operators import QubitHamiltonian
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
        energy, _ = rotosolve(exp, ansatz.var_params_default, ansatz, qubit_hamiltonian, extrapolate=False)

        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_rotosolve_extrapolate(self):
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
        energy, _ = rotosolve(exp, ansatz.var_params_default, ansatz, qubit_hamiltonian, extrapolate=True)

        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)


    def test_rotoselect(self):
        """Test rotoselect using a single-qubit Euler rotation circuit"""

        sim = get_backend()

        # Build an Euler rotation circuit as an ansatz
        euler_circuit = Circuit([
            Gate('RZ', 0, parameter=0, is_variational=True),
            Gate('RX', 0, parameter=0, is_variational=True),
            Gate('RZ', 0, parameter=0, is_variational=True)
        ])
        ansatz = VariationalCircuitAnsatz(euler_circuit)

        # Build a single-qubit Hamiltonian
        hamiltonian = \
            QubitHamiltonian((0,'X'), 1.0) + \
            QubitHamiltonian((0,'Y'), 2,0) + \
            QubitHamiltonian((0,'Z'), 3.0)

        # Define function to calculate energy and update parameters and rotation axes
        def exp_rotoselect(var_params, var_rot_axes, ansatz, qubit_hamiltonian):
            ansatz.update_var_params(var_params)
            for i, axis in enumerate(var_rot_axes):
                ansatz.circuit._variational_gates[i].name = axis
            energy = sim.get_expectation_value(qubit_hamiltonian, ansatz.circuit)
            return energy

        # Run rotoselect, return energy, parameters and axes of rotation:
        init_params = ansatz.var_params_default
        init_axes = ['RX']*len(init_params)
        energy, _, axes = rotoselect(exp_rotoselect, 
                                init_params, init_axes, ansatz, hamiltonian)

        # compare with exact energy:
        min_energy = -np.sqrt(1**2 + 2**2 + 3**2)
        self.assertAlmostEqual(energy, min_energy, delta=1e-4)

        # Ensure axes are all valid rotation gates:
        self.assertTrue(set(axes).issubset({'RX', 'RY', 'RZ'}))


    def test_rotoselect_heisenberg(self):
        """Test rotoselect using the 5-qubit periodic Heisenberg model"""
        
        sim = get_backend()
        n_qubits = 3
        n_layers = 2
        J = h = 1.0

        # Construct a "hardware efficient" CZ-based ansatz layer
        heisenberg_gates = [Gate('Ry', i,parameter=0, is_variational=True) for i in range(n_qubits)]
        heisenberg_gates += [Gate('CZ', i, (i+1)%n_qubits) for i in range(0,n_qubits-1,2)]
        heisenberg_gates += [Gate('CZ', i, (i+1)%n_qubits) for i in range(1,n_qubits,2)]
        heisenberg_layer = Circuit(heisenberg_gates)

        heisenberg_circuit = Circuit()
        for _ in range(n_layers):
            heisenberg_circuit += heisenberg_layer
        ansatz = VariationalCircuitAnsatz(heisenberg_circuit)

        # Construct periodic Heisenberg Hamiltonian
        hamiltonian = QubitHamiltonian()
        for i in range(n_qubits):
            hamiltonian += QubitHamiltonian((i,'Z'), h)
            for S in ['X','Y','Z']:
                hamiltonian += QubitHamiltonian([(i,S),((i+1)%n_qubits,S)],J)

        # Define function to calculate energy and update parameters and rotation axes
        def exp_rotoselect(var_params, var_rot_axes, ansatz, qubit_hamiltonian):
            ansatz.update_var_params(var_params)
            for i, axis in enumerate(var_rot_axes):
                ansatz.circuit._variational_gates[i].name = axis
            energy = sim.get_expectation_value(qubit_hamiltonian, ansatz.circuit)
            return energy

        # Run rotoselect, return energy, parameters and axes of rotation:
        init_params = [np.pi/3]*ansatz.n_var_params
        init_axes = ['RX']*len(init_params)
        energy, _, axes = rotoselect(exp_rotoselect,
                                init_params, init_axes, ansatz, hamiltonian)

        # compare with known ground state energy:
        min_energy = -4.0
        self.assertAlmostEqual(energy, min_energy, delta=1e-4)

        # Ensure axes are all valid rotation gates:
        self.assertTrue(set(axes).issubset({'RX', 'RY', 'RZ'}))

if __name__ == "__main__":
    unittest.main()
