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

"""This module defines the reduced UCCs ansatz class (RUCC, refering to both
UCC1 and UCC3), providing the foundation to implement variational ansatz
circuits. They are UCCD and UCCSD ansatz, but terms acting in the same way on an
Hartree-Fock initial state have been removed.

This must be used on a 2 levels system (2 MOs, 4 SOs) to be physically relevant.

Reference for those circuits.
    - McCaskey, A.J., Parks, Z.P., Jakowski, J. et al.
        Quantum chemistry as a benchmark for near-term quantum computers.
        npj Quantum Inf 5, 99 (2019).
        https://doi.org/10.1038/s41534-019-0209-0
"""

import numpy as np

from tangelo.linq import Circuit, Gate

from .ansatz import Ansatz


class RUCC(Ansatz):
    """This class implements the reduced-UCC ansatz, i.e. UCC1=UCCD and
    UCC3=UCCSD. Currently, only closed-shell is supported. This implies that the
    mean-field is computed with the RHF reference integrals.

    Args:
        n_var_params (int): Number of variational parameters, must be 1 or 3.
    """

    def __init__(self, n_var_params=1):

        if n_var_params not in {1, 3}:
            raise ValueError("The number of parameters for RUCC must be 1 or 3.")

        self.n_var_params = n_var_params

        # Supported reference state initialization.
        self.supported_reference_state = {"HF"}
        # Supported variational params initialization.
        self.supported_initial_var_params = {"zeros", "ones", "random"}

        # Default initial parameters for initialization.
        self.var_params_default = "zeros"
        self.reference_state_initialization = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros, random numbers
        providing some keywords for users, and also supporting direct user input
        (list or numpy array). Return the parameters so that workflows such as
        VQE can retrieve these values.
        """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = np.random.random((self.n_var_params,))
            elif var_params == "zeros":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g.
        prepare reference wavefunction with HF, multi-reference state, etc).
        This method outputs |1010>.

        Returns:
            Circuit: |1010> initial state.
        """

        if self.reference_state_initialization not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        # NB: this one is consistent with JW but not other transforms.
        if self.reference_state_initialization == "HF":
            return Circuit([Gate("X", target=i) for i in (0, 2)])

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state
        preparation ansatz (with currently specified initial_state and
        var_params).

        Args:
            list: Initial variational parameters. Must be consistent with the
                chosen UCC (1 or 3).
        """

        # Set initial variational parameters used to build the circuit.
        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Prepare reference state circuit |1010>.
        reference_state_circuit = self.prepare_reference_state()

        # If self.n_var_params has been changed manually, last check before building the circuit.
        if self.n_var_params == 1:
            rucc_circuit = self._ucc1()
        elif self.n_var_params == 3:
            rucc_circuit = self._ucc3()
        else:
            raise ValueError("The number of parameters for RUCC must be 1 or 3.")

        self.circuit = reference_state_circuit + rucc_circuit
        self.update_var_params(self.var_params)

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the already-built
        ansatz circuit member. The circuit does not need to be rebuilt every
        time if only the variational parameters change.

        Args:
            list: Variational parameters to parse into the circuit.
        """

        assert len(var_params) == self.n_var_params
        self.var_params = var_params

        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]

    def _ucc1(self):
        """This class implements the reduced-UCC ansatz UCC1. UCC1 is equivalent
        to the UCCD ansatz, but terms that act in the same manner of the HF
        state are removed.

        Returns:
            Circuit: UCC1 quantum circuit.
        """

        # Initialization of an empty list.
        lst_gates = list()

        # UCC1 gates are appended.
        lst_gates += [Gate("RX", 0, parameter=np.pi/2)]

        lst_gates += [Gate("H", qubit_i) for qubit_i in range(1, 4)]
        lst_gates += [Gate("CNOT", qubit_i+1, qubit_i) for qubit_i in range(3)]

        # Rz with var param to modifiy double excitation.
        lst_gates += [Gate("RZ", 3, parameter="theta", is_variational=True)]

        lst_gates += [Gate("CNOT", qubit_i, qubit_i-1) for qubit_i in range(3, 0, -1)]
        lst_gates += [Gate("H", qubit_i) for qubit_i in range(3, 0, -1)]

        lst_gates += [Gate("RX", 0, parameter=-np.pi/2)]

        return Circuit(lst_gates)

    def _ucc3(self):
        """This class implements the reduced-UCC ansatz UCC3. UCC3 is equivalent
        to the UCCSD ansatz, but terms that act in the same manner of the HF
        state are removed.

        Returns:
            Circuit: UCC3 quantum circuit.
        """

        # Initialization of an empty list.
        lst_gates = list()

        # UCC3 gates are appended.
        lst_gates += [Gate("RX", 0, parameter=np.pi/2)]
        lst_gates += [Gate("H", 1)]
        lst_gates += [Gate("RX", 2, parameter=np.pi/2)]
        lst_gates += [Gate("H", 3)]

        lst_gates += [Gate("CNOT", qubit_i, qubit_i-1) for qubit_i in (1, 3)]

        # Rz with var param to modifiy single excitations.
        lst_gates += [Gate("RZ", 1, parameter="theta0", is_variational=True)]
        lst_gates += [Gate("RZ", 3, parameter="theta1", is_variational=True)]

        lst_gates += [Gate("CNOT", 3, 2)]

        lst_gates += [Gate("RX", 2, parameter=-np.pi/2)]
        lst_gates += [Gate("H", 2)]

        lst_gates += [Gate("CNOT", qubit_i, qubit_i-1) for qubit_i in (2, 3)]

        # Rz with var param to modifiy double excitation.
        lst_gates += [Gate("RZ", 3, parameter="theta2", is_variational=True)]

        lst_gates += [Gate("CNOT", qubit_i, qubit_i-1) for qubit_i in range(3, 0, -1)]
        lst_gates += [Gate("H", qubit_i) for qubit_i in range(3, 0, -1)]

        lst_gates += [Gate("RX", 0, parameter=-np.pi/2)]

        return Circuit(lst_gates)
