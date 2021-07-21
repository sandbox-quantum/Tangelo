""" This module defines the k-UpCCGSD ansatz class. It provides a chemically inspired ansatzs
    and is an implementation of the classical unitary CCGSD ansatz. Generalized Single and pairwise
    double excitation determinants, in accordance with the system number of electron and spin, are considered.
    For more information about this ansatz, see references below.

    Refs:
        * Joonho Lee, William J. Huggins, Martin Head-Gordon, and K. Birgitta, "Generalized Unitary
          Couple Cluster Wavefunctions for Quantum Computation" arxiv:1810.02327
"""

import numpy as np

from agnostic_simulator import Circuit

from .ansatz import Ansatz
from .ansatz_utils import pauliword_to_circuit
from ._paired_unitary_cc import get_upccgsd
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit


class UpCCGSD(Ansatz):
    """ This class implements the UpCCGSD ansatz.
        This implies that the mean-field is computed with the RHF or ROHF reference integrals.
        Args:
            molecule (MolecularData) : the molecular system
            mean-field (optional) : mean-field of molecular system
                                    Default, None
            qubit_mapping (str) : one of the supported qubit mapping identifiers
                                  Default, 'jw'
            up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down
                                 Default, False (i.e. has alternating spin up/down ordering)
            k : parameters for the number of times UpCCGSD is repeated see (arxiv:1810.02327) for details
                Default, 2
    """

    def __init__(self, molecule, ansatz_options=dict()):
        default_options = {"qubit_mapping": 'jw', "mean_field": None, "up_then_down": False,
                           "k": 2}

        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in ansatz_options.items():
            if k in default_options:
                default_options[k] = v
            else:
                raise KeyError(f"Keyword :: {k}, not available in VQESolver")

        self.molecule = molecule
        # Write default options
        for k, v in default_options.items():
            setattr(self, k, v)

        self.mf = self.mean_field  # necessary duplication for get_rdm in vqe_solver

        # Later: refactor to handle various flavors of UCCSD
        if molecule.n_qubits % 2 != 0:
            raise ValueError('The total number of spin-orbitals should be even.')
        self.n_spatial_orbitals = self.molecule.n_qubits // 2
        self.n_spin_orbitals = self.molecule.n_qubits
        self.spin = self.molecule.multiplicity - 1
        self.n_doubles = self.n_spatial_orbitals * (self.n_spatial_orbitals - 1)//2
        self.n_singles = 2*self.n_doubles
        self.n_var_params_per_step = self.n_doubles + self.n_singles
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
                initial_var_params = 1.e-1 * (np.random.random((self.n_var_params,)) - 0.5)
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
                                         mapping=self.qubit_mapping,
                                         up_then_down=self.up_then_down,
                                         spin=self.spin)

    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Prepare reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Build qubit operator required to build k-UpCCGSD

        # Build dictionary of ordered pauli words for fast update of parameters intead of rebuilding circuit
        # Initialize dictionary of, dictionaries for each UpCCGSD step (current_k)
        self.pauli_to_angles_mapping = dict()
        # Initialize array of integers to keep track of starting point in qubit_op for each UpCCGSD step (current_k)
        sum_prev_qubit_terms = np.zeros(self.k+1, dtype=int)

        pauli_words_gates = list()
        for current_k in range(self.k):
            qubit_op = self._get_qubit(current_k)
            pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))

            # Initialize dictionary of qubit_op terms for each UpCCGSD step
            self.pauli_to_angles_mapping[current_k] = dict()

            # Obtain quantum circuit through trivial trotterization of the qubit operator for each current_k
            for i, (pauli_word, coef) in enumerate(pauli_words):
                pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
                self.pauli_to_angles_mapping[current_k][pauli_word] = i + sum_prev_qubit_terms[current_k]

            sum_prev_qubit_terms[current_k + 1] = len(qubit_op.terms.items())

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

        # Loop through each current_k step
        for current_k in range(self.k):
            # Build qubit operator required to build UpCCGSD for current_k
            qubit_op = self._get_qubit(current_k)

            # If qubit operator terms have changed, rebuild circuit. Else, simply update variational gates directly
            if set(self.pauli_to_angles_mapping[current_k].keys()) != set(qubit_op.terms.keys()):
                self.build_circuit(var_params)
                break
            else:
                for pauli_word, coef in qubit_op.terms.items():
                    gate_index = self.pauli_to_angles_mapping[current_k][pauli_word]
                    self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def _get_qubit(self, current_k):
        """Construct UpCCGSD FermionOperator for current_k variational parameters, and translate to QubitOperator
        via relevant qubit mapping.

        Returns:
            qubit_op (QubitOperator): qubit-encoded elements of the UpCCGSD ansatz for current_k.
        """
        current_k_params = self.var_params[current_k*self.n_var_params_per_step:(current_k+1)*self.n_var_params_per_step]

        fermion_op = get_upccgsd(self.n_spatial_orbitals, current_k_params)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.qubit_mapping,
                                            n_spinorbitals=self.molecule.n_qubits,
                                            n_electrons=self.molecule.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op
