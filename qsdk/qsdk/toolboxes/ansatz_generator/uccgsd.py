"""This module defines the adaptive UCCGSD ansatz class. """

import numpy as np

from agnostic_simulator import Circuit
from numpy.core.fromnumeric import var
from qsdk.toolboxes.operators.operators import FermionOperator
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.ansatz_generator.ansatz_utils import pauliword_to_circuit
from qsdk.toolboxes.ansatz_generator.ansatz import Ansatz
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.ansatz_generator._general_unitary_cc import uccgsd_generator, get_singles_number, get_doubles_number


class UCCGSD(Ansatz):
    """TBD
    """

    def __init__(self, n_spinorbitals, n_electrons, mapping='jw', up_then_down=False):

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.mapping = mapping
        self.up_then_down = up_then_down

        self.n_singles = get_singles_number(n_spinorbitals // 2)
        self.n_doubles = get_doubles_number(n_spinorbitals // 2)
        self.n_var_params = self.n_singles + self.n_doubles

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random"}

        # Default initial parameters for initialization
        self.var_params_default = "ones"
        self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, MP2 (...), providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = np.random.random((self.n_var_params,))
        else:
            try:
                assert (len(var_params) == self.n_var_params)
                initial_var_params = np.array(var_params)
            except AssertionError:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but received {len(var_params)}.")
        self.var_params = initial_var_params
        return initial_var_params

    def update_var_params(self, var_params):
        """ Shortcut: set value of variational parameters in the already-built ansatz circuit member.
            Preferable to rebuilt your circuit from scratch, which can be an involved process. """
        print(var_params)
        self.set_var_params(var_params)

        # Build qubit operator required to build UCCSD
        qubit_op = self.get_generators()

        # If qubit operator terms have changed, rebuild circuit. Else, simply update variational gates directly
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def prepare_reference_state(self):
        """ Prepare a circuit generating the HF reference state. """

        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping, up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build qubit operator required to build UCCSD
        qubit_op = self.get_generators()

        # Prepend reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Obtain quantum circuit through trivial trotterization of the qubit operator
        # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        pauli_words_gates = []
        self.pauli_to_angles_mapping = dict()
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i

        uccsd_circuit = Circuit(pauli_words_gates)
        # skip over the reference state circuit if it is empty
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + uccsd_circuit
        else:
            self.circuit = uccsd_circuit

    def get_generators(self):
        """TBD
        """

        # Initialize pool of (qubit) operators like in ADAPT-VQE paper, based on single and double excitations.
        lst_fermion_op = uccgsd_generator(self.n_spinorbitals, single_coeffs=self.var_params[:self.n_singles], double_coeffs=self.var_params[self.n_singles:])
        fermion_op = FermionOperator()
        for f_op in lst_fermion_op:
            fermion_op += f_op

        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.mapping,
                                            n_spinorbitals=self.n_spinorbitals,
                                            n_electrons=self.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op
