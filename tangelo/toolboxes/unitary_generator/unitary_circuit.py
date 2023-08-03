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

from typing import Union, List

import numpy as np

from tangelo.toolboxes.unitary_generator import Unitary
from tangelo.linq import Circuit


class CircuitUnitary(Unitary):
    """Class that implements the necessary methods for QPE given a Circuit that represents a Unitary."""

    def __init__(self, circuit: Circuit, control_method: str = "all"):
        """
        Args:
            circuit (Circuit): The circuit to apply the evolution to
            control_method (str): The method to add controlled gates to the circuit.
                "all" adds the control qubits to all gates
                "variational" only adds the control qubits to the gates marked as is_variationl
        """
        self.circuit = circuit
        self.valid_step_methods = ["all", "variational"]
        if control_method in self.valid_step_methods:
            self.n_steps_method = control_method
        else:
            raise ValueError(f"{self.__class__.__name__} only supports {self.valid_control_methods} to apply the control to the circuit.")
        self.state_qubits = self.circuit.width
        self.ancilla_qubits = []

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
            control

        Returns:
            Circuit: The circuit that implements the unitary evolution for n_steps.
        """
        if not method:
            method = self.n_steps_method

        return self.add_controls(method, control)*n_steps

    def add_controls(self, method: str = "all", control: Union[int, List[int], None] = None):
        """Adds control gates to the circuit"""
        new_circuit = self.circuit.copy()

        if control is None:
            return new_circuit
        if type(control) is list and len(control) > 0:
            clist = control
        elif type(control, (int, np.integer)):
            clist = [control]
        if method == "all":
            for gate in new_circuit:
                if gate.name[0] == "C":
                    gate.control += clist
                else:
                    gate.name = "C" + gate.name
                    gate.control = clist
        elif method == "variational":
            for gate in new_circuit._variational_gates:
                if gate.name[0] == "C":
                    gate.control += clist
                else:
                    gate.name = "C" + gate.name
                    gate.control = clist
        else:
            raise ValueError(f"{self.__class__.__name__} only supports {self.valid_control_methods} to apply the control to the circuit.")
