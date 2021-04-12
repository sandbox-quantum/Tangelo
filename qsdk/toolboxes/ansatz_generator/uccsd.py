""" This module defines the ansatz abstract class, providing the foundation to implement variational ansatz circuits """

import itertools
import numpy as np
from pyscf import mp

from agnostic_simulator import Circuit, Gate

from .ansatz import Ansatz
from .ansatz_utils import pauliword_to_circuit
from ._unitary_cc import uccsd_singlet_generator
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF


class UCCSD(Ansatz):
    """ This class implements the UCCSD ansatz. Currently, only closed-shell UCCSD is supported.
     This implies that the mean-field is computed with the RHF reference integrals. """

    def __init__(self, molecule, mapping='jw',mean_field=None):

        self.molecule = molecule
        self.mf = mean_field
        self.mapping = mapping

        # Later: refactor to handle various flavors of UCCSD
        if molecule.n_qubits % 2 != 0:
            raise ValueError('The total number of spin-orbitals should be even.')
        self.n_spatial_orbitals = self.molecule.n_qubits // 2
        self.n_occupied = int(np.ceil(self.molecule.n_electrons / 2))
        self.n_virtual = self.n_spatial_orbitals - self.n_occupied
        self.n_singles = self.n_occupied * self.n_virtual
        self.n_doubles = self.n_singles * (self.n_singles + 1) // 2
        self.n_var_params = self.n_singles + self.n_doubles

        # Supported reference state initialization
        # TODO: support for others and compatibility with the qubit mapping used
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "MP2"}

        # Default initial parameters for initialization
        self.var_params_default = "MP2"
        self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, MP2 (...), providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if isinstance(var_params, str) and (var_params not in self.supported_initial_var_params):
            raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
        if var_params is None:
            var_params = self.var_params_default

        if var_params == "ones":
            initial_var_params = np.ones((self.n_var_params,), dtype=float)
        elif var_params == "random":
            initial_var_params = np.random.random((self.n_var_params,), dtype=float)
        elif var_params == "MP2":
            initial_var_params = self._compute_mp2_params()
        else:
            try:
                assert (len(var_params) == self.n_var_params)
                initial_var_params = np.array(var_params)
            except AssertionError:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but received {len(var_params)}.")
        self.var_params = initial_var_params
        return initial_var_params

    # TODO: Possible initial states must be compatible with qubit transform used, add check for it later on
    def prepare_reference_state(self):
        """ Returns circuit preparing the reference state of the ansatz (e.g prepare reference wavefunction with HF,
        multi-reference state, etc). These preparations must be consistent with the transform used to obtain the
        qubit operator.
        """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.default_reference_state == "HF":
            return get_reference_circuit(self.molecule.n_qubits, self.molecule.n_electrons, mapping=self.mapping)
        

    def build_circuit(self, var_params=None, qubit_mapping='jw'):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        self.set_var_params(var_params)

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit()

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
        #skip over the reference state circuit if it is empty
        try:
            self.circuit = reference_state_circuit + uccsd_circuit
        except ValueError:
            self.circuit = uccsd_circuit

    def update_var_params(self, var_params):
        """ Shortcut: set value of variational parameters in the already-built ansatz circuit member.
            Preferable to rebuilt your circuit from scratch, which can be an involved process. """

        self.var_params = var_params

        # TODO: we should have a dedicated build function to this. We shouldnt rewrite it every time. Use qubit mapping wrapper too
        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit()

        # If qubit operator terms haven't changed, perform fast parameter update
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            print(f"Pauli words in qubit operator have changed, rebuilding UCCSD ansatz circuit.")
            self.build_circuit(var_params)
        else:
            # Directly update angles in variational gates without rebuilding the circuit
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def _get_singlet_qubit(self):
        """Construct UCCSD FermionOperator for current variational parameters, and translate to QubitOperator
        via relevant qubit mapping.

        Returns:
            qubit_op (QubitOperator): qubit-encoded elements of the UCCSD ansatz.
        """
        ferm_op = uccsd_singlet_generator(self.var_params, self.molecule.n_qubits, self.molecule.n_electrons)
        qubit_op = fermion_to_qubit_mapping(ferm_op, mapping=self.mapping, n_qubits=self.molecule.n_qubits, n_electrons=self.molecule.n_electrons)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op

    def _compute_mp2_params(self):
        """ Computes the MP2 initial variational parameters.
        Compute the initial variational parameters with PySCF MP2 calculation, and then reorders the elements
        into the QEMIST convention. MP2 only has doubles (T2) amplitudes, thus the single (T1) amplitudes are set to a
        small non-zero value and added. The ordering for QEMIST is single, double (diagonal), double (non-diagonal).

        Returns:
            list: The initial variational parameters (float64).
        """

        # If no mean-field was passed, compute it using the module to do so, for the right integrals
        if not self.mf:
            self.mf = prepare_mf_RHF(self.molecule.mol)

        mp2_fragment = mp.MP2(self.mf)
        mp2_fragment.verbose = 0
        mp2_correlation_energy, mp2_t2 = mp2_fragment.kernel()

        # Get singles amplitude. Just get "up" amplitude, since "down" should be the same
        singles = [2.e-5] * (self.n_virtual * self.n_occupied)
        # Get singles and doubles amplitudes associated with one spatial occupied-virtual pair
        doubles_1 = [-mp2_t2[q, q, p, p]/2. if (abs(-mp2_t2[q, q, p, p]/2.) > 1e-15) else 0.
                     for p, q in itertools.product(range(self.n_virtual), range(self.n_occupied))]
        # Get doubles amplitudes associated with two spatial occupied-virtual pairs
        doubles_2 = [-mp2_t2[q, s, p, r] for (p, q), (r, s)
                     in itertools.combinations(itertools.product(range(self.n_virtual), range(self.n_occupied)), 2)]

        return singles + doubles_1 + doubles_2
