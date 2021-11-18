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

import warnings
import numpy as np
from random import choice

from .ansatz import Ansatz
from tangelo.backendbuddy import Circuit
from .ansatz_utils import pauliword_to_circuit
from ._qubit_mf import initialize_qmf_state_from_hf_vec, get_qmf_circuit
from ._qubit_cc import construct_DIS
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number, fermion_to_qubit_mapping 

class QCC(Ansatz):
    """
        This class implements the qubit coupled cluster (QCC) ansatz. Currently, only closed-shell QCC is supported.
        This implies that the mean-field is computed with the RHF reference integrals.

        Args:
            molecule (SecondQuantizedMolecule) : The molecular system.
            mapping (str) : see mapping_transform.py for options 'JW' (Jordan Wigner),
                            or 'BK' (Bravyi Kitaev), or 'SCBK' (symmetry-conserving Bravyi Kitaev)
                Default, 'JW'
            up_then_down (bool): if True, change basis ordering putting all spin up orbitals first, followed by all spin down.
                Default, False 
            qcc_deriv_thresh (float): Threshold for the gradient magnitude |dEQCC/dtau| used to determine whether a particular direct interaction set (DIS) group
                                      of generators should be considered when forming the QCC operator.
                Default, 1.e-3 a.u.
            max_qcc_gens (int or None): Desired maximum number of generators to include in the QCC operator. If None, build the QCC operator
                                        with one generator from each DIS group characterized by |dEQCC/tau| > qcc_deriv_thresh. 
                                        If max_qcc_gens is an integer, then the QCC operator will be built with min(size(DIS), max_qcc_gens) generators.
                Default, None.                                        
            qubit_op_list (list of QubitOperators): List of generators to use for building the QCC operator instead of selecting
                                                    from DIS groups.
            qmf_var_params (list or numpy array of floats): QMF parameter set {Omega}. This set is required to build the DIS of QCC generators.
                                                            If None {Omega} can be initialized from a Hartree-Fock state.
                Default, None
            qmf_circuit (Circuit): QMF state preparation circuit. 
                Default, None
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, qcc_guess=1.e-1, qcc_deriv_thresh=1.e-3, max_qcc_gens=None, qubit_op_list=None, qmf_var_params=None, qmf_circuit=None):

        self.molecule = molecule
        self.mapping = mapping
        self.up_then_down = up_then_down

        if self.mapping.upper() == 'JW':
            if not self.up_then_down:
                warnings.warn(" If mapping == 'JW', the QCC ansatz requires spin-orbitals to be ordered all spin up followed by all spin down.", RuntimeWarning)
                self.up_then_down = True

        self.qcc_guess = qcc_guess
        self.qcc_deriv_thresh = qcc_deriv_thresh
        self.max_qcc_gens = max_qcc_gens
        self.qubit_op_list = qubit_op_list
        self.qmf_var_params = qmf_var_params
        self.qmf_circuit = qmf_circuit

        self.n_spinorbitals = self.molecule.n_active_sos
        self.n_electrons = self.molecule.n_active_electrons
        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)

        if self.n_qubits % 2 != 0:
            raise ValueError('The total number of spin-orbitals should be even.')

        self.fermi_ham = self.molecule.fermionic_hamiltonian
        self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,
                                                  n_spinorbitals=self.n_spinorbitals,
                                                  n_electrons=self.n_electrons,
                                                  up_then_down=self.up_then_down)

        if self.qmf_var_params is None:
            self.qmf_var_params = initialize_qmf_state_from_hf_vec(self.n_spinorbitals, self.n_electrons, self.mapping, self.up_then_down) 
        elif isinstance(self.qmf_var_params, list):
            self.qmf_var_params = np.array(self.qmf_var_params)

        if self.qmf_var_params.size != 2 * self.n_qubits:
            raise ValueError('The number of QMF variational parameters must be 2 * n_qubits.')           


        # Build the DIS of QCC generators and determine the number of QCC variational parameters.
        self.DIS = construct_DIS(self.qmf_var_params, self.qubit_ham, self.qcc_deriv_thresh)
        if self.qubit_op_list is None: 
            self.n_var_params = len(self.DIS) if self.max_qcc_gens is None else min(len(self.DIS), self.max_qcc_gens)
        else:
            self.n_var_params = len(self.qubit_op_list)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "random"}
        self.var_params = None

        # Default starting parameters for initialization
        self.default_reference_state = "HF"
        self.var_params_default = "random"
        self.qcc_circuit = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            var_params = var_params.lower()
            if var_params not in self.supported_initial_var_params:
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "zeros":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            elif var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = 2. * self.qcc_guess * np.random.random((self.n_var_params,)) - self.qcc_guess
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
            return get_qmf_circuit(self.qmf_var_params, variational=False)

    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build a qubit operator required for QCC 
        qubit_op = self._get_qcc_qubit_op()
  
        # Build a QMF state preparation circuit
        if self.qmf_circuit is None: 
            self.qmf_circuit = self.prepare_reference_state()
 
        # Obtain quantum circuit through trivial trotterization of the qubit operator
        # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        pauli_words_gates = list()
        self.pauli_to_angles_mapping = dict()
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i

        self.qcc_circuit = Circuit(pauli_words_gates)
        self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size is not None else self.qcc_circuit

    def update_var_params(self, var_params):
        """ 
        Shortcut: set value of variational parameters in the already-built ansatz circuit member.
        Preferable to rebuilding your circuit from scratch, which can be an involved process.
        """

        self.set_var_params(var_params)

        # Build a qubit operator required for QCC 
        qubit_op = self._get_qcc_qubit_op()

        # If qubit_op terms have changed, rebuild circuit. Else, simply update variational gates directly
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                self.qcc_circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef
            self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size is not None else self.qcc_circuit
        
    def _get_qcc_qubit_op(self):
        """
        Select one generator from the n_var_params unique DIS groups that were characterized by the largest gradient magnitudes.
        The QCC qubit operator is formed as the linear combination of those generators with var_params as coefficients.

        Args:
            var_params (numpy array of floats): QCC variational parameter set {tau}.
            DIS (list of lists): The direct interaction set of QCC generators. The DIS holds lists for each DIS group. Each DIS
                                 group list contains a list of all possible Pauli words for given its defining flip index and the 
                                 magnitude of its characteristic gradient dEQCC/dtau.
            n_var_params (int): number of generators to include in the QCC operator.
            qubit_op_list (list of QubitOperators): List of generators to use for building the QCC operator instead of selecting 
                                                    from DIS groups.
        Returns:
            qcc_qubit_op (QubitOperator): a linear combination of QCC generators that specifies the QCC operator:
                                          qcc_qubit_op = -0.5 * SUM_k P_k * tau_k,
        """

        print(' The QCC operator comprises {:} generator(s):\n'.format(self.n_var_params))
        qcc_qubit_op = QubitOperator.zero()
        # Build a QCC operator using either the DIS or a list of supplied generators
        if self.qubit_op_list is None:
            self.qubit_op_list = list()
            for i in range(self.n_var_params):
                # Randomly select a QCC generator from each DIS group.
                qcc_gen = choice(self.DIS[i][0])
                qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen
                self.qubit_op_list.append(qcc_gen)
                print('\tAmplitude for DIS group {:} generator {:} = {:} rad.\n'.format(i, str(qcc_gen), self.var_params[i]))
        else:
            try:
                assert (len(self.qubit_op_list) == self.n_var_params)
                for i, qcc_gen in enumerate(self.qubit_op_list):
                    qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen 
                    print('\tAmplitude for generator {:} = {:} rad.\n'.format(str(qcc_gen), self.var_params[i]))
            except AssertionError:
                raise ValueError(f"Expected {self.n_var_params} QubitOperators in self.qubit_op_list but received {len(self.qubit_op_list)}.")

        return qcc_qubit_op

