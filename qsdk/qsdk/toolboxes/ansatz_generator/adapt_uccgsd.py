"""Doc string"""

import math
import numpy as np

from agnostic_simulator import Circuit
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import pauliword_to_circuit
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.ansatz_generator._unitary_cc import uccsd_singlet_generator


class ADAPTUCCGSD(Ansatz):
    """Doc string """

    def __init__(self, n_spinorbitals, n_electrons, operators=list(), ferm_operators=list(), mapping="jw", up_then_down=False):

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.mapping = mapping
        self.up_then_down = up_then_down

        self.var_params = None
        self.circuit = None

        # The remaining of the constructor is useful to restart an ADAPT calculation.
        self.operators = operators
        self.length_operators = [len(pauli_op.terms) for pauli_op in operators]

        # Getting the sign of each pauli words. As UCCSD terms are hermitian,
        # each tau_i = excitation_i - deexcitation_i. The coefficient are
        # initialized to 1 (or -1).
        self.var_params_prefactor = list()
        for pauli_op in operators:
            for pauli_term in pauli_op.get_operators():
                coeff = list(pauli_term.terms.values())[0]
                self.var_params_prefactor += [math.copysign(1., coeff)]

        # Useful to keep track of excitations term, but not necessary.
        self.ferm_operators = ferm_operators

    @property
    def n_var_params(self):
        return len(self.length_operators)

    def set_var_params(self, var_params=None):
        """ Set initial variational parameter values. Defaults to zeros. """
        if var_params is None:
            var_params = [0.]*self.n_var_params
        elif len(var_params) != self.n_var_params:
            raise ValueError('Invalid number of parameters.')
        self.var_params = var_params
        return var_params

    def update_var_params(self, var_params):
        """ Update variational parameters (done repeatedly during VQE) """
        for var_index in range(self.n_var_params):
            length_op = self.length_operators[var_index] # 2 or 8

            param_index = sum(self.length_operators[:var_index])
            for param_subindex in range(length_op):
                prefactor = self.var_params_prefactor[param_index+param_subindex]
                self.circuit._variational_gates[param_index+param_subindex].parameter = prefactor*var_params[var_index]

    def prepare_reference_state(self):
        """ Prepare a circuit generating the HF reference state. """
        # TODO non hardcoded mapping
        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping, up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        """ Construct the variational circuit to be used as our ansatz. """
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
        """Add a new operator to our circuit"""

        self.length_operators += [len(pauli_operator.terms)]
        self.operators.append(pauli_operator)
        if ferm_operator is not None:
            self.ferm_operators += [ferm_operator]

        for pauli_term in pauli_operator.get_operators():
            coeff = list(pauli_term.terms.values())[0]
            self.var_params_prefactor += [math.copysign(1., coeff)]

            pauli_tuple = list(pauli_term.terms.keys())[0]
            new_operator = Circuit(pauliword_to_circuit(pauli_tuple, 0.1))

            self.circuit += new_operator

    def get_pool(self):
        """TBD
        """
        # Initialize pool of operators like in ADAPT-VQE paper, based on single and double excitations (Work in progress)
        # Use uccsd functions from openfermion to get a hold on the single and second excitations, per Lee's advice.

        n_spatial_orbitals = self.n_spinorbitals // 2
        n_occupied = int(np.ceil(self.n_electrons / 2))
        n_virtual = n_spatial_orbitals - n_occupied
        n_singles = n_occupied * n_virtual
        n_doubles = n_singles * (n_singles + 1) // 2
        n_amplitudes = n_singles + n_doubles

        f_op = uccsd_singlet_generator([1.]*n_amplitudes, 2*n_spatial_orbitals, self.n_electrons)
        lst_fermion_op = list()
        for i in f_op.get_operator_groups(len(f_op.terms) // 2):
            lst_fermion_op.append(i)

        pool_operators = [fermion_to_qubit_mapping(fermion_operator=fi,
                                                mapping=self.mapping,
                                                n_spinorbitals=self.n_spinorbitals,
                                                n_electrons=self.n_electrons,
                                                up_then_down=self.up_then_down) for fi in lst_fermion_op]

        # Cast all coefs to floats (rotations angles are real)
        for qubit_op in pool_operators:
            for key in qubit_op.terms:
                qubit_op.terms[key] = math.copysign(1., float(qubit_op.terms[key].imag))
            qubit_op.compress()

        return pool_operators, lst_fermion_op
