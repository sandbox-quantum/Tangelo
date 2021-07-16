""" This module defines the k-UpCCGSD ansatz class. It provides a chemically inspired ansatzs
    and is an implementation of the classical unitary GCCSD ansatz. Generalized Single and pairwise
    double excitation determinants, in accordance with the system number of electron and spin, are considered.
    For more information about this ansatz, see references below.

    Refs:
        * Joonho Lee, William J. Huggins, Martin Head-Gordon, and K. Birgitta, "Generalized Unitary
          Couple Cluster Wavefunctions for Quantum Computation" arxiv:1810.02327
"""

import itertools
import numpy as np
from pyscf import mp

from agnostic_simulator import Circuit

from .ansatz import Ansatz
from .ansatz_utils import pauliword_to_circuit
from ._paired_unitary_cc import get_upccgsd
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF


class UpCCGSD(Ansatz):
    """ This class implements the UpCCGSD ansatz.
     This implies that the mean-field is computed with the RHF or ROHF reference integrals. """

    def __init__(self, molecule, mapping='jw', mean_field=None, up_then_down=False, k=3):

        self.molecule = molecule
        self.k = k
        self.mf = mean_field
        self.mapping = mapping
        self.up_then_down = up_then_down

        # Later: refactor to handle various flavors of UCCSD
        if molecule.n_qubits % 2 != 0:
            raise ValueError('The total number of spin-orbitals should be even.')
        self.n_spatial_orbitals = self.molecule.n_qubits // 2
        self.n_spin_orbitals = self.molecule.n_qubits
        self.spin = self.molecule.multiplicity - 1
        self.n_doubles = self.n_spatial_orbitals * (self.n_spatial_orbitals - 1)//2
        self.n_singles = 2*self.n_doubles
        self.n_var_params_per_step = self.n_doubles + self.n_singles
        print(f'n per step {self.n_var_params_per_step}')
        print(self.molecule.n_electrons)
        self.n_var_params = self.k * (self.n_singles + self.n_doubles)

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.var_params_default = 'random'
        self.supported_initial_var_params = {"ones", "random"}

        # Default initial parameters for initialization
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

    def prepare_reference_state(self):
        """ Returns circuit preparing the reference state of the ansatz (e.g prepare reference wavefunction with HF,
        multi-reference state, etc). These preparations must be consistent with the transform used to obtain the
        qubit operator.
        """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.default_reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.molecule.n_qubits,
                                         n_electrons=self.molecule.n_electrons,
                                         mapping=self.mapping,
                                         up_then_down=self.up_then_down,
                                         spin=self.spin)

    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Prepend reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Build qubit operator required to build UpCCGSD
        pauli_words_gates = []
        self.pauli_to_angles_mapping = dict()
        sum_prev_qubit_terms = np.zeros(self.k+1, dtype=int)
        for current_k in range(self.k):
            qubit_op = self._get_qubit(current_k)
            self.pauli_to_angles_mapping[current_k] = dict()
            # Obtain quantum circuit through trivial trotterization of the qubit operator
            # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
            pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
            sum_prev_qubit_terms[current_k + 1] = len(qubit_op.terms.items())
            for i, (pauli_word, coef) in enumerate(pauli_words):
                pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
                self.pauli_to_angles_mapping[current_k][pauli_word] = i + sum_prev_qubit_terms[current_k]

        upccgsd_circuit = Circuit(pauli_words_gates)
        # skip over the reference state circuit if it is empty
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + upccgsd_circuit
        else:
            self.circuit = upccgsd_circuit

    def update_var_params(self, var_params):
        """ Shortcut: set value of variational parameters in the already-built ansatz circuit member.
            Preferable to rebuilt your circuit from scratch, which can be an involved process. """

        self.set_var_params(var_params)

        # Build qubit operator required to build UpCCGSD
        for current_k in range(self.k):
            qubit_op = self._get_qubit(current_k)

            # If qubit operator terms have changed, rebuild circuit. Else, simply update variational gates directly
            if set(self.pauli_to_angles_mapping[current_k].keys()) != set(qubit_op.terms.keys()):
                self.build_circuit(var_params)
            else:
                for pauli_word, coef in qubit_op.terms.items():
                    gate_index = self.pauli_to_angles_mapping[current_k][pauli_word]
                    self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def _get_qubit(self, current_k):
        """Construct k-UpCCGSD FermionOperator for current variational parameters, and translate to QubitOperator
        via relevant qubit mapping.

        Returns:
            qubit_op (QubitOperator): qubit-encoded elements of the UCCSD ansatz.
        """
        current_k_params = self.var_params[current_k*self.n_var_params_per_step:(current_k+1)*self.n_var_params_per_step]
        # print(f'curren_k {current_k}')
        # print(current_k_params)
        fermion_op = get_upccgsd(self.n_spatial_orbitals, current_k_params)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.mapping,
                                            n_spinorbitals=self.molecule.n_qubits,
                                            n_electrons=self.molecule.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op
