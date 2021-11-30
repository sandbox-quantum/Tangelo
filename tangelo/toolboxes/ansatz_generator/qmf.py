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

"""This module defines the qubit mean field (QMF) ansatz class. The ansatz is a
variational product state built from a set of parameterized single-qubit states.
For applications in quantum chemistry, the ansatz can be used to describe the
mean field component of an electronic wave function on a quantum computer. For
this reason, it is the foundation of the qubit coupled cluster (QCC)
method, which addresses the electron correlation component of an electronic wave
function that is neglected by the QMF ansatz. For more information about this
ansatz, see references below.

Refs:
    1. I. G. Ryabinkin, T.-C. Yen, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2018, 14 (12), 6317-6326.
    2. I. G. Ryabinkin and S. N. Genin.
        https://arxiv.org/abs/1812.09812 2018.
    3. S. N. Genin, I. G. Ryabinkin, and A. F. Izmaylov.
          https://arxiv.org/abs/1901.04715 2019.
    4. I. G. Ryabinkin, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2019, 15, 1, 249–255.
"""

import warnings
import numpy as np

from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number,\
    fermion_to_qubit_mapping
from .ansatz import Ansatz
from ._qubit_mf import get_qmf_circuit, init_qmf_state_from_hf_vec,\
    penalize_mf_ham


class QMF(Ansatz):
    """This class implements the QMF ansatz. Closed-shell and restricted open-shell QMF are
    supported. While the form of the QMF ansatz is the same for either variation, the underlying
    fermionic mean field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell QMF implies that spin = 0 or spin != 0 and the fermionic mean field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Optimizing QMF variational parameters is risky, epsecially when a random initial guess is used.
    It is recommended that penalty terms be addded to the mean field Hamiltonian to enforce
    appropriate electron number and spin angular momentum symmetries on the QMF wave function
    during optimization (see Ref. 4). If using penalty terms is to be avoided, an inital guess
    based on a Hartree-Fock reference state will likely converge quickly to the desired state, but
    it is not guaranteed.

    Args:
        molecule (SecondQuantizedMolecule) : The molecular system.
        mapping (str) : One of the supported qubit mapping identifiers. Default, "JW".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False (i.e. has alternating spin up/down ordering).
        qmf_state_init (str or dict): The QMF variational parameter set {Omega} procedure.
            If a dict, the mean field Hamiltonian is penalized using N, S^2, or Sz operators.
            The keys are (str) "init_params", "N", "S^2", and "Sz". The key "init_params" takes one
            of the values in self.supported_initial_var_params. For keys "N", "S^2", and "Sz",
            the values are tuples of the penatly term coefficient and the target value of the
            penalty operator: value=(mu, target). Keys and values are case sensitive and mu > 0.

            If "auto_pen" (str), a dict is automatically built with details for penalizing the mean
            field Hamiltonian with the N, S^2, and Sz operators. {Omega} values are initialized
            from a HF reference state vector. The target values for each penalty operator are
            derived from self.molecule as <N> = n_electrons, <Sz> = spin//2, and
            <S^2> = (spin//2)*(spin//2 + 1). The coefficient mu is set to 1. for all penalty terms.

            If a str in self.supported_initial_var_params, {Omega} is initialized according to the
            option definition in set_var_params. If the "hf-state" option is chosen, penalty terms
            are not used. For all other options, a dict is created as is done for "auto_pen".
            Default, "auto_pen".
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, qmf_state_init="auto_pen"):

        self.molecule = molecule
        self.n_spinorbitals = self.molecule.n_active_sos
        self.n_orbitals = self.n_spinorbitals // 2
        self.n_electrons = self.molecule.n_active_electrons
        self.spin = molecule.spin
        self.mapping = mapping
        self.up_then_down = up_then_down
        self.qmf_state_init = qmf_state_init

        if self.mapping.upper() == "JW":
            if not self.up_then_down:
                warn_msg = "If using the QMF and QCC ansatz classes together, set up_then_down "\
                           " = True when mapping = JW when initializing both classes."
                warnings.warn(warn_msg, RuntimeWarning)

        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)
        if self.n_qubits % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")

        self.n_var_params = 2 * self.n_qubits

        # get fermionic Hamiltonians and check if penalty terms will be added
        self.fermi_ham = self.molecule.fermionic_hamiltonian
        if isinstance(self.qmf_state_init, str):
            if self.qmf_state_init == "hf-state":
                self.var_params_default = "hf-state"
            else:
                if self.qmf_state_init == "auto_pen":
                    self.var_params_default = "hf-state"
                else:
                    self.var_params_default = self.qmf_state_init

                spin_z = self.spin // 2
                pen_details = {"N": (1.5, self.n_electrons), "S^2": (1.5, spin_z * (spin_z + 1)),\
                    "Sz": (1.5, spin_z)}

                self.fermi_ham += penalize_mf_ham(pen_details, self.n_orbitals,\
                    self.n_electrons, up_then_down=False)
        elif isinstance(self.qmf_state_init, dict):
            try:
                self.var_params_default = qmf_state_init["init_params"]
                self.fermi_ham += penalize_mf_ham(self.qmf_state_init, self.n_orbitals,\
                    self.n_electrons, up_then_down=False)
            except KeyError as k_err:
                raise KeyError("Key 'init_params' was not found in qmf_state_init.") from k_err
        else:
            raise TypeError("Unrecognized type self.qmf_state_init: must be str or dict.")

        self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,\
            n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons,\
            up_then_down=self.up_then_down, spin=self.spin)

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "pi_over_two", "random", "hf-state"}

        # Default starting parameters for initialization
        self.default_reference_state = "HF"
        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros, random numbers,
        or a Hartree-Fock state occupation vector, providing some keywords
        for users, and also supporting direct user input (list or numpy array).
        Return the parameters so that workflows such as VQE can retrieve these values. """

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
            elif var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "pi_over_two":
                initial_var_params = 0.5 * np.pi * np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_thetas = np.pi * np.random.random((self.n_qubits,))
                initial_phis = 2. * np.pi * np.random.random((self.n_qubits,))
                initial_var_params = np.concatenate((initial_thetas, initial_phis))
            elif var_params == "hf-state":
                initial_var_params = init_qmf_state_from_hf_vec(self.n_spinorbitals,\
                    self.n_electrons, self.mapping, up_then_down=self.up_then_down, spin=self.spin)
        else:
            try:
                assert len(var_params) == self.n_var_params
                initial_var_params = np.array(var_params)
            except AssertionError as as_err:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but\
                                   received {len(var_params)}.") from as_err
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g prepare reference
        wavefunction with HF, multi-reference state, etc). These preparations must be consistent
        with the transform used to obtain the qubit operator. """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:\
                               {self.supported_reference_state}")
        if self.default_reference_state == "HF":
            reference_state_circuit = get_qmf_circuit(self.var_params, variational=True)
        return reference_state_circuit

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params). """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build the circuit for the QMF ansatz
        self.circuit = self.prepare_reference_state()

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the already-built ansatz circuit member.
        Preferable to rebuilt your circuit from scratch, which can be an involved process. """

        # Rebuild the circuit because it is trivial
        self.build_circuit(var_params)

