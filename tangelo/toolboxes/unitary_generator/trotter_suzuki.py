# Copyright 2023 Good Chemistry Company.
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

from tangelo.toolboxes.unitary_generator import Unitary
from tangelo.toolboxes.operators import count_qubits, QubitOperator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize


class TrotterSuzukiUnitary(Unitary):
    """Class that implements the Trotter-Suzuki time evolution."""

    def __init__(self, qubit_hamiltonian: QubitOperator, time: float = 1., trotter_order: int = 1,
                 n_trotter_steps: int = 1, n_steps_method: str = "time"):
        """
        Args:
            qubit_hamiltonian (QubitOperator): The operator to time-evolve.
            time (float): The total time evolution
            trotter_order (int): The order of the Trotter-Suzuki time-evolution.
            n_trotter_steps (int): The number of Trotter steps to make for each time-evolution
            n_steps_method (str): Method to apply unitary multiple steps.
                "time" to not change the circuit size, less accurate
                "repeat" to repate the circuit n_steps
        """
        self.qubit_hamiltonian = qubit_hamiltonian
        self.time = time
        self.trotter_order = trotter_order
        self.n_trotter_steps = n_trotter_steps
        self.state_qubits = list(range(count_qubits(qubit_hamiltonian)))
        self.ancilla_qubits = []
        self.valid_step_methods = ["time", "repeat"]
        if n_steps_method in self.valid_step_methods:
            self.n_steps_method = n_steps_method
        else:
            raise ValueError(f"{self.__class__.__name__} only supports {self.valid_step_methods} to apply the unitary multiple times.")

    def qubit_indices(self):
        """Return the indices used by the algorithm to propagate the unitary.

        Returns:
            List[int]: State qubits
            List[int]: Ancilla qubits
        """
        return self.state_qubits, self.ancilla_qubits

    def build_circuit(self, n_steps: int, control: int = None, method: str = ""):
        """Build and return the quantum circuit implementing the unitary evolution for n_steps.

        Args:
            n_steps(int): The number of unitary evolution steps
            control (Union[int, List[int]]): The qubit or qubits to control with.
            method (str): The method used to apply the controlled operation for n_steps.
                "time" to not change the circuit size, less accurate
                "repeat" to repate the circuit n_steps

        Returns:
            Circuit: The circuit that implements the unitary evolution for n_steps with control.
        """
        if not method:
            method = self.n_steps_method

        if method == "time":
            return trotterize(self.qubit_hamiltonian, self.time*n_steps, self.n_trotter_steps, self.trotter_order, control=control)
        elif method == "repeat":
            return trotterize(self.qubit_hamiltonian, self.time, self.n_trotter_steps, self.trotter_order, control=control)*n_steps
        else:
            raise ValueError(f"{self.__class__.__name__} only supports {self.valid_step_methods} to apply the unitary multiple times.")
