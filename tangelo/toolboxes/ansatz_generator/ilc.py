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

"""This module defines the qubit coupled cluster ansatz class for involutory 
linear combinations (ILC) of anti-commuting Pauli products.

Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
"""

import warnings
import numpy as np

from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number,\
                                                               fermion_to_qubit_mapping
from tangelo.linq import Circuit
from .ansatz import Ansatz
from .ansatz_utils import exp_pauliword_to_gates
from ._qubit_mf import init_qmf_from_hf, get_qmf_circuit, purify_qmf_state
from ._qubit_ilc import construct_ACS, init_ilc_diag
from ._qubit_cc import construct_dis


class ILC(Ansatz):
    """This class implements the ILC ansatz. Closed-shell and restricted open-shell ILC are
    supported. While the form of the ILC ansatz is the same for either variation, the underlying
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell ILC implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "JW".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False.
        qubit_op_list (list of QubitOperator): A list of ILC generators to use for the ansatz.
            Default, None.
        qmf_circuit (Circuit): An instance of tangelo.linq Circuit class implementing a QMF state
            preparation circuit. A variational circuit can be passed from the QMF ansatz class.
            Otherwise a non-variational circuit is created by default. Default, None.
        qmf_var_params (list or numpy array of float): The QMF variational parameter set.
            If None, the values are determined using a Hartree-Fock reference state. Default, None.
        qubit_mf_ham (QubitOperator): Allows a qubit Hamiltonian to be passed to the ILC ansatz
            class. If not None, then the fermionic Hamiltonian is ignored. Default, None.
        ilc_guess (float): Sets the initial guess for all amplitudes in the ILC variational
            parameter set. Default, 1.e-1 a.u.
        ilc_deriv_thresh (float): Threshold value of |dEILC/dtau| so that if |dEILC/dtau| >=
            ilc_deriv_thresh for a generator, add its candidate group to the DIS.
        max_ilc_gens (int or None): Maximum number of generators allowed for the ansatz.
            If None, one generator from each DIS group is used. If set to an int, then
            min(size(DIS), max_ilc_gens) generators are used for the ansatz. Default, None.
        verbose (bool): Flag for ILC verbosity. Default, False.
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, qubit_op_list=None,
                 qmf_circuit=None, qmf_var_params=None, qubit_mf_ham=None, ilc_guess=1.e-1,
                 ilc_deriv_thresh=1.e-3, max_ilc_gens=None, n_trot=1, verbose=False):

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
            warnings.warn("The ILC ansatz requires spin-orbital ordering to be all spin-up "
                          "first followed by all spin-down for the JW mapping.", RuntimeWarning)
            self.up_then_down = True

        self.ilc_guess = ilc_guess
        self.ilc_deriv_thresh = ilc_deriv_thresh
        self.max_ilc_gens = max_ilc_gens
        self.n_trot = n_trot
        self.qubit_op_list = qubit_op_list
        self.qmf_var_params = qmf_var_params
        self.qmf_circuit = qmf_circuit
        self.verbose = verbose

        if qubit_mf_ham is None:
            self.fermi_ham = self.molecule.fermionic_hamiltonian
            self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,
                                                      self.n_spinorbitals, self.n_electrons,
                                                      self.up_then_down, self.spin)
        else:
            self.qubit_ham = qubit_mf_ham

        if self.qmf_var_params is None:
            self.qmf_var_params = init_qmf_from_hf(self.n_spinorbitals, self.n_electrons,
                                                   self.mapping, self.up_then_down, self.spin)
        elif isinstance(self.qmf_var_params, list):
            self.qmf_var_params = np.array(self.qmf_var_params)
        if self.qmf_var_params.size != 2 * self.n_qubits:
            raise ValueError("The number of QMF variational parameters must be 2 * n_qubits.")

        # Get purified QMF parameters and use them to build the DIS or use a list of generators.
        if self.qubit_op_list is None:
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals,
                                               self.n_electrons, self.mapping, self.up_then_down,
                                               self.spin, self.verbose)
            self.dis = construct_dis(pure_var_params, self.qubit_ham, self.ilc_deriv_thresh,
                                     self.verbose)
            self.acs = construct_acs(self.dis, self.max_ilc_gens, self.n_qubits)
            self.n_var_params = len(self.acs) if self.max_ilc_gens is None\
                                else min(len(self.acs), self.max_ilc_gens)
        else:
            self.dis = None
            self.n_var_params = len(self.qubit_op_list)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "diag", "ilc_guess"}

        # Default starting parameters for initialization
        self.pauli_to_angles_mapping = {}
        self.default_reference_state = "HF"
        self.var_params_default = "diag"
        self.var_params = None
        self.rebuild_dis = False
        self.ilc_circuit = None
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
            # Initialize the ILC wave function as |ILC> = |QMF>
            if var_params == "zeros":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            # Initialize all tau parameters to the same value specified by self.ilc_guess
            elif var_params == "diag":
                initial_var_params = init_ilc_diag(self.qubit_ham, self.acs, self.qmf_var_params)
            elif var_params == "ilc_guess":
                initial_var_params = self.ilc_guess * np.ones((self.n_var_params,))
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

        # Build a QMF state preparation circuit
        if self.qmf_circuit is None:
            self.qmf_circuit = self.prepare_reference_state()

        # Build a qubit operator required for ILC
        self._get_ilc_qubit_op()

        # Obtain quantum circuit through trivial trotterization of the qubit operator
        pauli_word_gates = []
        for i in range(self.n_trot):
            for j, ilc_op in enumerate(self.qubit_op_list):
                pauli_word, coef = list(ilc_op.terms.items())[0]
                pauli_word_gates += pauliword_to_circuit(pauli_word, float(coef/self.n_trot), variational=True)
        self.ilc_circuit = Circuit(pauli_word_gates)
        self.circuit = self.qmf_circuit + self.ilc_circuit if self.qmf_circuit.size != 0\
                       else self.ilc_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the existing ansatz circuit member.
        Preferable to rebuilding your circuit from scratch, which can be an involved process.
        """

        self.set_var_params(var_params)

        # Build the qubit operator required for ILC
        self._get_ilc_qubit_op()

        for pauli_word, coef in qubit_op.terms.items():
            gate_index = self.pauli_to_angles_mapping[pauli_word]
            gate_param = 2. * coef if coef >= 0. else 4 * np.pi + 2 * coef
            self.ilc_circuit._variational_gates[gate_index].parameter = gate_param
        self.circuit = self.qmf_circuit + self.ilc_circuit if self.qmf_circuit.size != 0\
                       else self.ilc_circuit

    def _get_ilc_qubit_op(self):
        """Returns the ILC operator by selecting one generator from n_var_params DIS groups.
        The ILC qubit operator is constructed as a linear combination of generators using the
        parameter set {tau} as coefficients: ILC operator = -0.5 * SUM_k P_k * tau_k.
        The exponentiated terms of the ILC operator, U = PROD_k exp(-0.5j * tau_k * P_k),
        are used to build a ILC circuit.

        Args:
            var_params (numpy array of float): The ILC variational parameter set.
            n_var_params (int): Size of the ILC variational parameter set.
            qmf_var_params (numpy array of float): The QMF variational parameter set.
            qubit_ham (QubitOperator): A qubit Hamiltonian.
            ilc_deriv_thresh (float): Threshold value of |dEILC/dtau| so that if |dEILC/dtau| >=
                ilc_deriv_thresh for a generator, add its candidate group to the DIS.
            dis (list of list): The DIS of ILC generators.
            acs (list of list): The ACS of DIS ILC operators.
            qubit_op_list (list of QubitOperator): A list of generators to use when building the ILC
                operator instead of selecting from DIS groups.
            rebuild_dis (bool): Rebuild the DIS. This is useful if qubit_ham of qmf_var_params have
                changed (e.g. in iterative methods like iILC or QCC-ILC). If True, qubit_op_list is
                reset to None.
            verbose (bool): Flag for ILC verbosity. Default, False.

        Returns:
            None
        """

        # Rebuild DIS & ACS in case qubit_ham changed or they and qubit_op_list don't exist
        if self.rebuild_dis or (self.dis is None and self.qubit_op_list is None):
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals,
                                               self.n_electrons, self.mapping, self.up_then_down,
                                               self.spin, self.verbose)
            self.dis = construct_dis(pure_var_params, self.qubit_ham, self.ilc_deriv_thresh,
                                     self.verbose)
            self.acs = construct_acs(self.dis, self.max_ilc_gens, self.n_qubits)
            self.n_var_params = len(self.acs) if self.max_ilc_gens is None\
                                else min(len(self.acs), self.max_ilc_gens)
            self.qubit_op_list = None

        # Build the ILC qubit operator list 
        self.qubit_op_list = []
        for i in range(self.n_var_params - 1, 1, -1):
            self.qubit_op_list.append(-0.5 * self.var_params[i - 1] * self.acs[i])
        self.qubit_op_list.append(-1. * self.var_params[0] * self.acs[1])
        for i in range(2, self.n_var_params):
            self.qubit_op_list.append(-0.5 * self.var_params[i - 1] * self.acs[i])
