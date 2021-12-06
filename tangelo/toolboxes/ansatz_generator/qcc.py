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
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.linq import Circuit

from .ansatz import Ansatz
from .ansatz_utils import pauliword_to_circuit
from ._qubit_mf import init_qmf_state_from_hf_vec, get_qmf_circuit
from ._qubit_cc import construct_dis


class QCC(Ansatz):
    """This class implements the QCC ansatz. Closed-shell and restricted open-shell QCC are
    supported. While the form of the QCC ansatz is the same for either variation, the underlying
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell QCC implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "JW".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False (i.e. has alternating spin up/down ordering).
        qcc_guess (float): Sets the initial guess for all amplitudes in the QCC variational
            parameter set {tau}. Default, 1.e-1 a.u.
        qcc_deriv_thresh (float): Threshold of the value of |dEQCC/dtau| for a generator from
            a candidate DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group
            enters the DIS and its generators can be used in the QCC ansatz.
        max_qcc_gens (int or None): Maximum number of generators to include in the QCC operator.
            If None, build the QCC operator with one generator from each DIS group characterized by
            |dEQCC/tau| >= qcc_deriv_thresh. If max_qcc_gens is an int, then use
            min(size(DIS), max_qcc_gens) generators to build the QCC operator. Default, None.
        qubit_op_list (list of QubitOperator): A list of generators to use when building the QCC
            operator instead of selecting from DIS groups.
        qubit_mf_ham (QubitOperator): Allows a qubit Hamiltonian to be passed to the QCC ansatz
            class during initilization. This enables straightforward construction of the DIS
            with a user-specified Hamiltonian (e.g. a penalized mean-field qubit Hamiltonian from
            a mean-field simulation with the QMF ansatz). If not None, then the fermionic
            Hamiltonian in molecule is ignored. Default, None.
        qmf_var_params (list or numpy array of float): The QMF parameter variational set {Omega}.
            If None, {Omega} is determined using a reference state Hartree-Fock occupation vector.
            Default, None.
        qmf_circuit (Circuit): A QMF state preparation circuit. Enables a QMF circuit to be
            passed with parameters that can be treated variationally or not (e.g. a circuit
            prepared by the QMF ansatz class). If None, then one is built using qmf_var_params
            and the parameters are not treated variationally (i.e. the QMF state is fixed).
            Default, None.
        verbose (bool): Flag for QCC verbosity. Default, False.
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, qcc_guess=1.e-1,\
        qcc_deriv_thresh=1.e-3, max_qcc_gens=None, qubit_op_list=None, qubit_mf_ham=None,\
        qmf_var_params=None, qmf_circuit=None, verbose=False):

        self.molecule = molecule
        self.n_spinorbitals = self.molecule.n_active_sos

        if self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")

        self.n_electrons = self.molecule.n_active_electrons
        self.spin = molecule.spin
        self.mapping = mapping
        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)
        self.up_then_down = up_then_down

        if self.mapping.upper() == "JW" and not self.up_then_down:
            warn_msg = "The QCC ansatz requires spin-orbital ordering to be all spin-up "\
                       "first followed by all spin-down for the JW mapping."
            warnings.warn(warn_msg, RuntimeWarning)
            self.up_then_down = True

        self.qcc_guess = qcc_guess
        self.qcc_deriv_thresh = qcc_deriv_thresh
        self.max_qcc_gens = max_qcc_gens
        self.qubit_op_list = qubit_op_list
        self.qmf_var_params = qmf_var_params
        self.qmf_circuit = qmf_circuit
        self.verbose = verbose

        if qubit_mf_ham is None:
            self.fermi_ham = self.molecule.fermionic_hamiltonian
            self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,\
                n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons,\
                up_then_down=self.up_then_down, spin=self.spin)
        else:
            self.qubit_ham = qubit_mf_ham

        if self.qmf_var_params is None:
            self.qmf_var_params = init_qmf_state_from_hf_vec(self.n_spinorbitals,\
                self.n_electrons, self.mapping, up_then_down=self.up_then_down, spin=self.spin)
        elif isinstance(self.qmf_var_params, list):
            self.qmf_var_params = np.array(self.qmf_var_params)

        if self.qmf_var_params.size != 2 * self.n_qubits:
            raise ValueError("The number of QMF variational parameters must be 2 * n_qubits.")

        # Build the DIS from scratch or use a list of generators.
        if self.qubit_op_list is None:
            self.dis = construct_dis(self.qmf_var_params, self.qubit_ham,\
                self.qcc_deriv_thresh, verbose=self.verbose)
            self.n_var_params = len(self.dis) if self.max_qcc_gens is None\
                else min(len(self.dis), self.max_qcc_gens)
        else:
            self.dis = None
            self.n_var_params = len(self.qubit_op_list)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "qcc_guess"}

        # Default starting parameters for initialization
        self.pauli_to_angles_mapping = {}
        self.default_reference_state = "HF"
        self.var_params_default = "qcc_guess"
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
                err_msg = f"Supported keywords for initializing variational parameters: "\
                          f"{self.supported_initial_var_params}"
                raise ValueError(err_msg)
            if var_params == "zeros":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            elif var_params == "qcc_guess":
                initial_var_params = self.qcc_guess * np.ones((self.n_var_params,))
        elif np.array(var_params).size == self.n_var_params:
            initial_var_params = np.array(var_params)
        elif np.array(var_params).size != self.n_var_params:
            err_msg = f"Expected {self.n_var_params} variational parameters but "\
                      f"received {np.array(var_params).size}."
            raise ValueError(err_msg)
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g prepare reference
        wavefunction with HF, multi-reference state, etc). These preparations must be consistent
        with the transform used to obtain the qubit operator. """

        if self.default_reference_state not in self.supported_reference_state:
            err_msg = f"Only supported reference state methods are: "\
                      f"{self.supported_reference_state}."
            raise ValueError(err_msg)
        if self.default_reference_state == "HF":
            reference_state_circuit = get_qmf_circuit(self.qmf_var_params, variational=False)
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
        # Keep track of the order in which pauli words have been visited for fast parameter updates
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        pauli_words_gates = []
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i

        self.qcc_circuit = Circuit(pauli_words_gates)
        self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size != 0\
            else self.qcc_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the existing ansatz circuit member.
        Preferable to rebuilding your circuit from scratch, which can be an involved process.
        """

        self.set_var_params(var_params)

        # Build the qubit operator required for QCC
        qubit_op = self._get_qcc_qubit_op()

        # If qubit_op terms have changed, rebuild circuit. else update variational gates directly
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                gate_param = 2.*coef if coef >= 0. else 4*np.pi+2*coef
                self.qcc_circuit._variational_gates[gate_index].parameter = gate_param
            self.circuit = self.qmf_circuit + self.qcc_circuit if self.qmf_circuit.size != 0\
                else self.qcc_circuit

    def _get_qcc_qubit_op(self):
        """Returns the QCC operator by selecting one generator from n_var_params DIS groups.
        The QCC qubit operator is constructed as a linear combination of generators using the
        parameter set {tau} as coefficients: QCC operator = -0.5 * SUM_k P_k * tau_k.
        The exponentiated terms of the QCC operator, U = PROD_k exp(-0.5j * tau_k * P_k),
        are used to build a QCC circuit.

        Args:
            var_params (numpy array of float): The QCC variational parameter set {tau}.
            n_var_params (int): Size of the QCC variational parameter set.
            qmf_var_params (numpy array of float): The QMF variational parameter set {Omega}.
            qubit_ham (QubitOperator): A qubit Hamiltonian.
            qcc_deriv_thresh (float): Threshold of the value of |dEQCC/dtau| for a generator from
                a candidate DIS group. If |dEQCC/dtau| >= qcc_deriv_thresh, the candidate DIS group
                enters the DIS and its generators can be used in the QCC ansatz.
            dis (list of list): The DIS of QCC generators. Each list in dis contains (1) a complete
                set of generators for a DIS group built from Pauli X and an odd number of Y
                operators that act on qubits indexed by all combinations of the flip indices and
                (2) the signed value of dEQCC/dtau.
            qubit_op_list (list of QubitOperator): A list of generators to use when building the QCC
                operator instead of selecting from DIS groups.
            rebuild_dis (bool): Rebuild the DIS. This is useful if qubit_ham of qmf_var_params have
                changed (e.g. in iterative methods like iQCC or QCC-ILC). If True, qubit_op_list is
                reset to None.
            verbose (bool): Flag for QCC verbosity. Default, False.

        Returns:
            QubitOperator: QCC ansatz qubit operator.
        """

        # Rebuild the DIS in case qubit_ham changed or both the DIS and qubit_op_list don't exist
        if self.rebuild_dis or (self.dis is None and self.qubit_op_list is None):
            self.dis = construct_dis(self.qmf_var_params, self.qubit_ham, self.qcc_deriv_thresh,\
                verbose=self.verbose)
            self.n_var_params = len(self.dis) if self.max_qcc_gens is None\
                else min(len(self.dis), self.max_qcc_gens)
            self.qubit_op_list = None

        # Build the QCC operator using the DIS or a list of generators
        qcc_qubit_op = QubitOperator.zero()
        if self.qubit_op_list is None:
            self.qubit_op_list = []
            for i in range(self.n_var_params):
                dis_group = self.dis[i]
                # Instead of randomly choosing a generator, get the last one.
                qcc_gen = dis_group[0][-1]
                qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen
                self.qubit_op_list.append(qcc_gen)
        else:
            if len(self.qubit_op_list) == self.n_var_params:
                for i, qcc_gen in enumerate(self.qubit_op_list):
                    qcc_qubit_op -= 0.5 * self.var_params[i] * qcc_gen
            else:
                err_msg = f"Expected {self.n_var_params} generators in {self.qubit_op_list} but "\
                          f"received {len(self.qubit_op_list)}.\n"
                raise ValueError(err_msg)
        return qcc_qubit_op
