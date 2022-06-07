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

"""This module defines the adaptive ansatz class."""

import math

from tangelo.linq import Circuit
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


class ADAPTAnsatz(Ansatz):
    """Adaptive ansatz used with ADAPT-VQE. This Ansatz class has methods to
    take a (or many) QubitOperator, transform it (them) into a circuit and
    append it (them). The number of variational parameters corresponds to the
    number of terms added to the Ansatz.

    Attributes:
        n_spinorbitals (int): Number of spin orbitals in a given basis.
        n_electrons (int): Number of electrons.
        operators (list of QubitOperator): List of operators to consider at the
            construction step. Can be useful for restarting computation.
        ferm_operators (list of FermionOperator): Same as operators, but in
            fermionic form. Not necessarily for running the ansatz, but it is
            convenient for analyzing results.
        mapping (string): Qubit encoding.
        up_then_down (bool): Ordering convention.
        var_params (list of float): Variational parameters.
        circuit (Circuit): Quantum circuit defined by a list of Gates.
    """

    def __init__(self, n_spinorbitals, n_electrons, ansatz_options=None):
        default_options = {"operators": list(), "ferm_operators": list(),
                           "mapping": "jw", "up_then_down": False,
                           "reference_state": "HF"}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        if ansatz_options:
            for k, v in ansatz_options.items():
                if k in default_options:
                    setattr(self, k, v)
                else:
                    raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons

        self.var_params = None
        self.circuit = None

        # The remaining of the constructor is useful to restart an ADAPT calculation.
        self._n_terms_operators = [len(pauli_op.terms) for pauli_op in self.operators]

        # Getting the sign of each pauli words. As UCCSD terms are hermitian,
        # each tau_i = excitation_i - deexcitation_i. The coefficient are
        # initialized to 1 (or -1).
        self._var_params_prefactor = list()
        for pauli_op in self.operators:
            for pauli_term in pauli_op.get_operators():
                coeff = list(pauli_term.terms.values())[0]
                self._var_params_prefactor += [math.copysign(1., coeff)]

    @property
    def n_var_params(self):
        return len(self._n_terms_operators)

    def set_var_params(self, var_params=None):
        """Set initial variational parameter values. Defaults to zeros."""

        if var_params is None:
            var_params = [0.]*self.n_var_params
        elif len(var_params) != self.n_var_params:
            raise ValueError('Invalid number of parameters.')
        self.var_params = var_params

        return var_params

    def update_var_params(self, var_params):
        """Update variational parameters (done repeatedly during VQE)."""

        for var_index in range(self.n_var_params):
            length_op = self._n_terms_operators[var_index]

            param_index = sum(self._n_terms_operators[:var_index])
            for param_subindex in range(length_op):
                prefactor = self._var_params_prefactor[param_index+param_subindex]
                self.circuit._variational_gates[param_index+param_subindex].parameter = prefactor*var_params[var_index]

    def prepare_reference_state(self):
        """Prepare a circuit generating the HF reference state."""
        if self.reference_state.upper() == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping, up_then_down=self.up_then_down)
        else:
            return Circuit(n_qubits=self.n_spinorbitals)

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""

        self.set_var_params(var_params)

        self.circuit = Circuit(n_qubits=self.n_spinorbitals)
        self.circuit += self.prepare_reference_state()
        adapt_circuit = list()
        for op in self.operators:
            for pauli_term in op.get_operators():
                pauli_tuple = list(pauli_term.terms.keys())[0]
                adapt_circuit += exp_pauliword_to_gates(pauli_tuple, 0.1)

        adapt_circuit = Circuit(adapt_circuit)
        if adapt_circuit.size != 0:
            self.circuit += adapt_circuit

        # Must be included because after convergence, the circuit is rebuilt.
        # If this is not present, the gradient on the previous operator chosen
        # is selected again (as the parameters are not minimized in respect to it
        # as with initialized coefficient to 0.1). If the parameters are
        # updated, it stays consistent.
        self.update_var_params(self.var_params)

        return self.circuit

    def add_operator(self, pauli_operator, ferm_operator=None):
        """Add a new operator to our circuit.

        Args:
            pauli_operator (QubitOperator): Operator to convert into a circuit
                and append it to the present circuit.
            ferm_operator (FermionicOperator): Same operator in fermionic form.
        """

        # Keeping track of the added operator.
        self._n_terms_operators += [len(pauli_operator.terms)]
        self.operators.append(pauli_operator)

        if ferm_operator is not None:
            self.ferm_operators += [ferm_operator]

        # Going through every term and convert them into a circuit. Keeping track
        # of each term sign.
        for pauli_term in pauli_operator.get_operators():
            coeff = list(pauli_term.terms.values())[0]
            self._var_params_prefactor += [math.copysign(1., coeff)]

            pauli_tuple = list(pauli_term.terms.keys())[0]
            new_operator = Circuit(exp_pauliword_to_gates(pauli_tuple, 0.1))

            self.circuit += new_operator
