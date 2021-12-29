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

""" This module defines an ansatz class to wrap up a custom tangelo.linq
    circuit.
"""

import numpy as np

from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz


class VariationalCircuitAnsatz(Ansatz):
    """This class takes an arbitrary circuit and convert it to an Ansatz. This
    enables users to provide a custom pre-built circuit.

    Args:
        Circuit: Circuit with variational gates.
    """

    def __init__(self, abstract_circuit):

        self.circuit = abstract_circuit

        self.n_var_params = len(self.circuit._variational_gates)
        self.var_params = None

        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "zeros"}

        # Default initial parameters for initialization
        self.var_params_default = [gate.parameter for gate in self.circuit._variational_gates]

    def set_var_params(self, var_params=None):
        """Set initial variational parameter values. Defaults to random."""

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            var_params = var_params.lower()

            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            else:
                if var_params == "ones":
                    var_params = np.ones((self.n_var_params,), dtype=float)
                elif var_params == "random":
                    var_params = 4 * np.pi * (np.random.random((self.n_var_params,)) - 0.5)
                elif var_params == "zeros":
                    var_params = np.zeros((self.n_var_params,), dtype=float)
        else:
            var_params = np.array(var_params)
            if var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {var_params.size}.")
        self.var_params = var_params
        return var_params

    def update_var_params(self, var_params):
        """Update variational parameters (done repeatedly during VQE)."""

        self.set_var_params(var_params)
        var_params = self.var_params

        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]

    def prepare_reference_state(self):
        """Method not needed as it is expected to be in the circuit provided."""
        pass

    def build_circuit(self, var_params=None):
        """Update parameters of the pre-built circuit."""

        self.set_var_params(var_params)
        self.update_var_params(self.var_params)

        return self.circuit
