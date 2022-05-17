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

"""This module defines the qubit coupled cluster (QCC) ansatz class. The
motivation behind this ansatz is to provide an improved alternative to
classical unitary coupled cluster for describing the electron correlation of
molecular systems. This implementation is based on Ref. 1, where the ansaztz
takes the form of a variational product state built directly from a set of
parameterized exponentiated qubit operators. A qubit operator is selected for
the ansatz based on an energy gradient criterion that indicates its potential
contribution to variational lowering of the QCC energy. For chemical applications,
the quantum mean-field ansatz is used in conjunction with this ansatz to
describe an electronic wave function on a quantum computer. For more information
about this ansatz and its variations, see references below.

Refs:
    1. I. G. Ryabinkin, T.-C. Yen, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2018, 14 (12), 6317-6326.
    2. I. G. Ryabinkin, R. A. Lang, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.
    3. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
"""

import warnings

import numpy as np

from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number,\
                                                               fermion_to_qubit_mapping
from tangelo.linq import Circuit
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.ansatz_generator._qubit_mf import init_qmf_from_hf, get_qmf_circuit, purify_qmf_state
from tangelo.toolboxes.ansatz_generator._qubit_cc import construct_dis


class QCC(Ansatz):
    """This class implements the QCC ansatz. Closed-shell and restricted open-shell QCC are
    supported. While the form of the QCC ansatz is the same for either variation, the underlying
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell QCC implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin-up orbitals first,
            followed by all spin-down. Default, False.
        qcc_op_list (list of QubitOperator): Generator list for the QCC ansatz. Default, None.
        qmf_circuit (Circuit): An instance of tangelo.linq Circuit class implementing a QMF state
            circuit. If passed from the QMF ansatz class, parameters are variational.
            If None, one is created with QMF parameters that are not variational. Default, None.
        qmf_var_params (list or numpy array of float): QMF variational parameter set.
            If None, the values are determined using a Hartree-Fock reference state. Default, None.
        qubit_ham (QubitOperator): Pass a qubit Hamiltonian to the QCC ansatz class and ignore
            the fermionic Hamiltonian in molecule. Default, None.
        deqcc_dtau_thresh (float): Threshold for |dEQCC/dtau| so that a candidate group is added
            to the DIS if |dEQCC/dtau| >= deqcc_dtau_thresh for a generator. Default, 1.e-3 a.u.
        qcc_tau_guess (float): The initial guess for all QCC variational parameters.
            Default, 1.e-2 a.u.
        max_qcc_gens (int or None): Maximum number of generators allowed in the ansatz. If None,
            one generator from each DIS group is selected. If int, then min(|DIS|, max_qcc_gens)
            generators are selected in order of decreasing |dEQCC/dtau|. Default, None.
    """

    def __init__(self, molecule, mapping="jw", up_then_down=False, qcc_op_list=None,
                 qmf_circuit=None, qmf_var_params=None, qubit_ham=None, qcc_tau_guess=1.e-2,
                 deqcc_dtau_thresh=1.e-3, max_qcc_gens=None):

        if not molecule:
            raise ValueError("An instance of SecondQuantizedMolecule is required for initializing "
                             "the QCC ansatz class.")
        self.molecule = molecule
        self.mapping = mapping
        self.up_then_down = up_then_down
        if self.mapping.lower() == "jw" and not self.up_then_down:
            warnings.warn("Efficient generator screening for the QCC ansatz requires spin-orbital "
                          "ordering to be all spin-up first followed by all spin-down for the JW "
                          "mapping.", RuntimeWarning)
            self.up_then_down = True

        self.n_spinorbitals = self.molecule.n_active_sos
        if self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")

        self.spin = molecule.spin
        self.fermi_ham = self.molecule.fermionic_hamiltonian
        self.n_electrons = self.molecule.n_electrons
        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)
      
        self.qubit_ham = qubit_ham
        if qubit_ham is None:
            self.fermi_ham = self.molecule.fermionic_hamiltonian
            self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,
                                                      self.n_spinorbitals, self.n_electrons,
                                                      self.up_then_down, self.spin)

        self.qmf_var_params = qmf_var_params
        if self.qmf_var_params is None:
            self.qmf_var_params = init_qmf_from_hf(self.n_spinorbitals, self.n_electrons,
                                                   self.mapping, self.up_then_down, self.spin)
        elif isinstance(self.qmf_var_params, list):
            self.qmf_var_params = np.array(self.qmf_var_params)
        if self.qmf_var_params.size != 2 * self.n_qubits:
            raise ValueError("The number of QMF variational parameters must be 2 * n_qubits.")

        self.qmf_circuit = qmf_circuit

        self.qcc_op_list = qcc_op_list
        self.qcc_tau_guess = qcc_tau_guess
        self.deqcc_dtau_thresh = deqcc_dtau_thresh
        self.max_qcc_gens = max_qcc_gens

        # Get purified QMF parameters and use them to build the DIS or use a list of generators.
        if self.qcc_op_list is None:
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals, self.n_electrons,
                                               self.mapping, self.up_then_down, self.spin)
            self.dis = construct_dis(self.qubit_ham, pure_var_params, self.deqcc_dtau_thresh)
            self.n_var_params = len(self.dis) if self.max_qcc_gens is None\
                                else min(len(self.dis), self.max_qcc_gens)
        else:
            self.dis = None
            self.n_var_params = len(self.qcc_op_list)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"qmf_state", "qcc_tau_guess", "random"}

        # Default starting parameters for initialization
        self.pauli_to_angles_mapping = {}
        self.default_reference_state = "HF"
        self.var_params_default = "random"
        self.var_params = None
        self.rebuild_dis = False
        self.qcc_circuit = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros or floats,
        providing some keywords for users, and also supporting direct user input
        (list or numpy array). Return the parameters so that workflows such as VQE can
        retrieve these values. """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            var_params = var_params.lower()
            if var_params not in self.supported_initial_var_params:
                raise ValueError(f"Supported keywords for initializing variational parameters: "
                                 f"{self.supported_initial_var_params}")
            # Initialize the QCC wave function as |QCC> = |QMF>
            if var_params == "qmf_state":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            # Initialize all tau parameters to the same value specified by self.qcc_tau_guess
            elif var_params == "qcc_tau_guess":
                initial_var_params = self.qcc_tau_guess * np.ones((self.n_var_params,))
            # Initialize tau parameters randomly over the domain [-qcc_tau_guess, qcc_tau_guess]
            elif var_params == "random":
                initial_var_params = 2. * self.qcc_tau_guess * np.random.random((self.n_var_params,)) - self.qcc_tau_guess
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g prepare reference
        wavefunction with HF, multi-reference state, etc). These preparations must be consistent
        with the transform used to obtain the qubit operator. """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are: "
                             f"{self.supported_reference_state}.")
        if self.default_reference_state == "HF":
            reference_state_circuit = get_qmf_circuit(self.qmf_var_params, False)
        return reference_state_circuit

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params). """

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
        # Track the order in which pauli words have been visited for fast parameter updates
        pauli_words_gates = []
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += exp_pauliword_to_gates(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i
        self.qcc_circuit = Circuit(pauli_words_gates)
        self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size != 0\
                       else self.qcc_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the existing ansatz circuit member.
        Preferable to rebuilding your circuit from scratch, which can be an involved process.
        """

        # Update the QCC variational parameters
        self.set_var_params(var_params)

        # Build the QCC ansatz qubit operator
        qubit_op = self._get_qcc_qubit_op()

        # If qubit_op terms have changed, rebuild circuit
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        # Otherwise update variational gates directly
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                gate_param = 2. * coef if coef >= 0. else 4 * np.pi + 2 * coef
                self.qcc_circuit._variational_gates[gate_index].parameter = gate_param
            self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size != 0\
                           else self.qcc_circuit

    def _get_qcc_qubit_op(self):
        """Returns the QCC operator by selecting one generator from n_var_params DIS groups.
        The QCC qubit operator is constructed as a linear combination of generators using the
        parameter set {tau} as coefficients: QCC operator = -0.5 * SUM_k P_k * tau_k.
        The exponentiated terms of the QCC operator, U = PROD_k exp(-0.5j * tau_k * P_k),
        are used to build a QCC circuit.

        Returns:
            QubitOperator: QCC ansatz qubit operator.
        """

        # Rebuild DIS if qubit_ham or qmf_var_params changed or if DIS and qcc_op_list are None.
        if self.rebuild_dis or (not self.dis and not self.qcc_op_list):
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals, self.n_electrons,
                                               self.mapping, self.up_then_down, self.spin)
            self.dis = construct_dis(self.qubit_ham, pure_var_params, self.deqcc_dtau_thresh)
            self.n_var_params = len(self.dis) if self.max_qcc_gens is None\
                                else min(len(self.dis), self.max_qcc_gens)
            self.qcc_op_list = None

        # Build the QCC operator using the DIS or a list of generators
        qcc_qubit_op = QubitOperator.zero()
        if self.qcc_op_list is None:
            self.qcc_op_list = []
            for i in range(self.n_var_params):
                # Instead of randomly choosing a generator, grab the first one.
                qcc_gen = self.dis[i][0]
                qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen
                self.qcc_op_list.append(qcc_gen)
        else:
            if len(self.qcc_op_list) == self.n_var_params:
                for i, qcc_gen in enumerate(self.qcc_op_list):
                    qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen
            else:
                raise ValueError(f"Expected {self.n_var_params} generators in "
                                 f"{self.qcc_op_list} but received {len(self.qcc_op_list)}.\n")
        return qcc_qubit_op
