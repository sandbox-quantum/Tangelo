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

"""This module defines the qubit coupled cluster ansatz class with involutory
linear combinations (ILC) of anticommuting sets (ACS) of Pauli words
(generators). Relative to the direct interation set (DIS) of QCC generators,
which incur an exponential growth of Hamiltonian terms upon dressing, the ACS
of ILC generators enables Hamiltonian dressing such that the number of terms
grows quadratically and exact quadratic truncation of the Baker-Campbell-Hausdorff
expansion. For more information about this ansatz, see references below.

Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        arXiv:2002.05701v1, 2020, 1–10.
    2. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
"""

import warnings

import numpy as np

from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number,\
                                                               fermion_to_qubit_mapping
from tangelo.linq import Circuit
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.ansatz_generator.ansatz_utils import exp_pauliword_to_gates
from tangelo.toolboxes.ansatz_generator._qubit_mf import init_qmf_from_hf, get_qmf_circuit, purify_qmf_state
from tangelo.toolboxes.ansatz_generator._qubit_cc import construct_dis
from tangelo.toolboxes.ansatz_generator._qubit_ilc import construct_acs, get_ilc_params_by_diag


class ILC(Ansatz):
    """This class implements the ILC ansatz. Closed-shell and restricted open-shell ILC are
    supported. While the form of the ILC ansatz is the same for either variation, the underlying
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell ILC implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported  mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin-up orbitals first,
            followed by all spin-down. Default, False.
        acs (list of QubitOperator): The mutually anticommuting generator list for the ILC ansatz.
            Default, None.
        qmf_circuit (Circuit): An instance of tangelo.linq Circuit class implementing a QMF state
            circuit. If passed from the QMF ansatz class, parameters are variational.
            If None, one is created with QMF parameters that are not variational. Default, None.
        qmf_var_params (list or numpy array of float): QMF variational parameter set.
            If None, the values are determined using a Hartree-Fock reference state. Default, None.
        qubit_ham (QubitOperator): Pass a qubit Hamiltonian to the  ansatz class and ignore
            the fermionic Hamiltonian in molecule. Default, None.
        deilc_dtau_thresh (float): Threshold for |dEILC/dtau| so that a candidate group is added
            to the DIS if |dEILC/dtau| >= deilc_dtau_thresh for a generator. Default, 1e-3 a.u.
        ilc_tau_guess (float): The initial guess for all ILC variational parameters.
            Default, 1e-2 a.u.
        max_ilc_gens (int or None): Maximum number of generators allowed in the ansatz. If None,
            one generator from each DIS group is selected. If int, then min(|DIS|, max_ilc_gens)
            generators are selected in order of decreasing |dEILC/dtau|. Default, None.
    """

    def __init__(self, molecule, mapping="jw", up_then_down=False, acs=None,
                 qmf_circuit=None, qmf_var_params=None, qubit_ham=None, ilc_tau_guess=1e-2,
                 deilc_dtau_thresh=1e-3, max_ilc_gens=None):

        if not molecule:
            raise ValueError("An instance of SecondQuantizedMolecule is required for initializing "
                             "the self.__class__.__name__ ansatz class.")
        self.molecule = molecule
        self.mapping = mapping
        self.up_then_down = up_then_down
        if self.mapping.lower() == "jw" and not self.up_then_down:
            warnings.warn("Spin-orbital ordering shifted to all spin-up first then down to "
                          "ensure efficient generator screening for the Jordan-Wigner mapping "
                          "with the self.__class__.__name__ ansatz.", RuntimeWarning)
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

        self.acs = acs
        self.ilc_tau_guess = ilc_tau_guess
        self.deilc_dtau_thresh = deilc_dtau_thresh
        self.max_ilc_gens = max_ilc_gens

        # Get purified QMF parameters and build the DIS & ACS or use a list of generators.
        if self.acs is None:
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals,
                                               self.n_electrons, self.mapping, self.up_then_down, self.spin)
            self.dis = construct_dis(self.qubit_ham, pure_var_params, self.deilc_dtau_thresh)
            self.max_ilc_gens = len(self.dis) if self.max_ilc_gens is None\
                                else min(len(self.dis), self.max_ilc_gens)
            self.acs = construct_acs(self.dis, self.max_ilc_gens, self.n_qubits)
            self.n_var_params = len(self.acs)
        else:
            self.dis = None
            self.n_var_params = len(self.acs)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"qmf_state", "ilc_tau_guess", "random", "diag"}

        # Default starting parameters for initialization
        self.default_reference_state = "HF"
        self.var_params_default = "diag"
        self.var_params = None
        self.rebuild_dis = False
        self.rebuild_acs = False
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
            if var_params == "qmf_state":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            # Initialize all ILC parameters to the same value specified by self.ilc_tau_guess
            elif var_params == "ilc_tau_guess":
                initial_var_params = self.ilc_tau_guess * np.ones((self.n_var_params,))
            # Initialize tau parameters randomly over the domain [0., 2 pi)
            elif var_params == "random":
                initial_var_params = 2. * np.pi * np.random.random((self.n_var_params,))
            # Initialize ILC parameters by matrix diagonalization (see Appendix B, Refs. 1 & 2).
            elif var_params == "diag":
                initial_var_params = get_ilc_params_by_diag(self.qubit_ham, self.acs, self.qmf_var_params)
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
        with the transform used to obtain the  operator. """

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

        # Build create the list of ILC qubit operators
        self.ilc_op_list = self._get_ilc_op()

        # Obtain quantum circuit through trotterization of the list of ILC operators
        pauli_word_gates = []
        for ilc_op in self.ilc_op_list:
            pauli_word, coef = list(ilc_op.terms.items())[0]
            pauli_word_gates += exp_pauliword_to_gates(pauli_word, coef, variational=True)
        self.ilc_circuit = Circuit(pauli_word_gates)
        self.circuit = self.qmf_circuit + self.ilc_circuit if self.qmf_circuit.size != 0\
                       else self.ilc_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the existing ansatz circuit member.
        Preferable to rebuilding your circuit from scratch, which can be an involved process.
        """

        # Update the ILC variational parameters
        self.set_var_params(var_params)

        # Build the ILC ansatz operator
        self.ilc_op_list = self._get_ilc_op()

        pauli_word_gates = []
        for ilc_op in self.ilc_op_list:
            pauli_word, coef = list(ilc_op.terms.items())[0]
            pauli_word_gates += exp_pauliword_to_gates(pauli_word, coef, variational=True)
        self.ilc_circuit = Circuit(pauli_word_gates)
        self.circuit = self.qmf_circuit + self.ilc_circuit if self.qmf_circuit.size != 0\
                       else self.ilc_circuit

    def _get_ilc_op(self):
        """Returns the ILC operators ordered according to the argument of
            Eq. C1, Appendix C, Ref. 1.

        Returns:
            list of QubitOperator: the list of ILC qubit operators
        """

        # Rebuild DIS & ACS in case qubit_ham changed or they and qubit_op_list don't exist
        if self.rebuild_dis or self.rebuild_acs or not self.acs:
            pure_var_params = purify_qmf_state(self.qmf_var_params, self.n_spinorbitals,
                                               self.n_electrons, self.mapping, self.up_then_down, self.spin)
            self.dis = construct_dis(self.qubit_ham, pure_var_params, self.deilc_dtau_thresh)
            self.max_ilc_gens = len(self.dis) if self.max_ilc_gens is None\
                                else min(len(self.dis), self.max_ilc_gens)
            self.acs = construct_acs(self.dis, self.max_ilc_gens, self.n_qubits)
            self.ilc_op_list = None

        # Build the ILC qubit operator list
        ilc_op_list = []
        for i in range(self.n_var_params - 1, 0, -1):
            ilc_op_list.append(-0.5 * self.var_params[i] * self.acs[i])
        ilc_op_list.append(-self.var_params[0] * self.acs[0])
        for i in range(1, self.n_var_params):
            ilc_op_list.append(-0.5 * self.var_params[i] * self.acs[i])
        return ilc_op_list
