"""This module defines the adaptive ansatz class. """

import math

from agnostic_simulator import Circuit
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import pauliword_to_circuit
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


class ADAPTAnsatz(Ansatz):
    """Adaptive ansatz used with ADAPT-VQE. It is agnostic in relation to the
    operator pool chosen. Compared to a normal Ansatz class, it can add a qubit
    operator and assign a single variational parameter to it. Therefore, the
    number of parameters is the amount of iterations where an operator is added.

    Args:
        ansatz_options (dict): ansatz options to defined attributes.

    Attributes:
        n_spinorbitals (int): Number of spin orbitals in a given basis.
        n_electrons (int): Number of electrons.
        operators (list of QubitOperator): List of operator to consider at the
            construction step. Can be useful for restarting computation.
        ferm_operators (list of FermionOperator): Same as operators, but in
            fermionic form. Not necessary for running the ansatz, but it is
            convenient for analyzing results.
        mapping (string): Qubit encoding.
        up_then_down (bool): Ordering convention.
        var_params (list of float): Variational parameters.
        circuit (angostic_simulation.Circuit): Quantum circuit.
        length_operators (list of int): Length of every self.operators. With an
            adaptive ansatz, one variational parameter correponds to a single
            cycle. Every term are hermitian, so there are many terms in each
            'meta' term. Therefore, many variational gates are defined for each
            operator in self.operators. This attributes is used to map the number
            of var_params to actual number of variational gates.
        var_params_prefactor (list of float): List of 1. or -1. to keep track of
            each sub-operators sign.
    """

    #def __init__(self, n_spinorbitals, n_electrons, pool_generator, operators=list(), ferm_operators=list(), mapping="jw", up_then_down=False):
    def __init__(self, ansatz_options=None):
        default_options = {"n_spinorbitals": 0, "n_electrons": 0,
                           "operators": list(), "ferm_operators":list(),
                           "mapping": "jw", "up_then_down": False,
                           "reference_state": "HF"}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in ansatz_options.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                # TODO Raise a warning instead, that variable will not be used unless user made mods to code
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        self.var_params = None
        self.circuit = None

        # The remaining of the constructor is useful to restart an ADAPT calculation.
        self.length_operators = [len(pauli_op.terms) for pauli_op in self.operators]

        # Getting the sign of each pauli words. As UCCSD terms are hermitian,
        # each tau_i = excitation_i - deexcitation_i. The coefficient are
        # initialized to 1 (or -1).
        self.var_params_prefactor = list()
        for pauli_op in self.operators:
            for pauli_term in pauli_op.get_operators():
                coeff = list(pauli_term.terms.values())[0]
                self.var_params_prefactor += [math.copysign(1., coeff)]

    @property
    def n_var_params(self):
        return len(self.length_operators)

    def set_var_params(self, var_params=None):
        """Set initial variational parameter values. Defaults to zeros. """

        if var_params is None:
            var_params = [0.]*self.n_var_params
        elif len(var_params) != self.n_var_params:
            raise ValueError('Invalid number of parameters.')
        self.var_params = var_params

        return var_params

    def update_var_params(self, var_params):
        """Update variational parameters (done repeatedly during VQE). """

        for var_index in range(self.n_var_params):
            length_op = self.length_operators[var_index]

            param_index = sum(self.length_operators[:var_index])
            for param_subindex in range(length_op):
                prefactor = self.var_params_prefactor[param_index+param_subindex]
                self.circuit._variational_gates[param_index+param_subindex].parameter = prefactor*var_params[var_index]

    def prepare_reference_state(self):
        """ Prepare a circuit generating the HF reference state. """

        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping, up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""

        self.set_var_params(var_params)

        self.circuit = Circuit(n_qubits=self.n_spinorbitals)
        self.circuit += self.prepare_reference_state()
        adapt_circuit = list()
        for op in self.operators:
            for pauli_term in op.get_operators():
                pauli_tuple = list(pauli_term.terms.keys())[0]
                adapt_circuit += pauliword_to_circuit(pauli_tuple, 0.1)

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
            pauli_operator (QubitOperator): Operator to convert into a circuit and
                append it to the present circuit.
            ferm_operator (FermionicOperator): Same operator in fermionic form.
        """

        # Keeping track of the added operator.
        self.length_operators += [len(pauli_operator.terms)]
        self.operators.append(pauli_operator)

        if ferm_operator is not None:
            self.ferm_operators += [ferm_operator]

        # Going through every term and convert them into a circuit. Keeping track
        # of each term sign.
        for pauli_term in pauli_operator.get_operators():
            coeff = list(pauli_term.terms.values())[0]
            self.var_params_prefactor += [math.copysign(1., coeff)]

            pauli_tuple = list(pauli_term.terms.keys())[0]
            new_operator = Circuit(pauliword_to_circuit(pauli_tuple, 0.1))

            self.circuit += new_operator
