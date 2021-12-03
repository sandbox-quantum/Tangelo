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

"""This module defines the qubit mean-field (QMF) ansatz class. The ansatz is a
variational product state built from a set of parameterized single-qubit states.
For applications in quantum chemistry, the ansatz can be used to describe the
mean-field component of an electronic wave function on a quantum computer.
For more information about this ansatz, see references below.

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
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell QMF implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Optimizing QMF variational parameters is risky, epsecially when a random initial guess is used.
    It is recommended that penalty terms be addded to the mean-field Hamiltonian to enforce
    appropriate electron number and spin angular momentum symmetries on the QMF wave function
    during optimization (see Ref. 4). If using penalty terms is to be avoided, an inital guess
    based on a Hartree-Fock reference state will likely converge quickly to the desired state, but
    it is not guaranteed.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "JW".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False (i.e. has alternating spin up/down
            ordering).
        init_qmf (dict): Controls the QMF variational parameter set {Omega} initialization
            procedure and mean-field Hamiltonian penalization. The keys are "init_params",
            "N", "S^2", or "Sz" (str). The value of "init_params" must be in
            supported_initial_var_params (str), which initializes {Omega} according to the
            option definition in set_var_params. The value of "N", "S^2", or "Sz" is (tuple
            or None). If a tuple, its elements are the penalty term coefficient (float) and
            the target value of a penalty operator (int), "key": (mu, target). If "N", "S^2",
            or "Sz" is None, a penalty term is added with default mu and target values:
            mu = 1.5 and target is derived from molecule as <N> = n_electrons,
            <S^2> = spin_z * (spin_z + 1), and <Sz> = spin_z, where spin_z = spin // 2. Key,
            value pairs are case sensitive and mu > 0. Default, {"init_params": "hf-state"}.
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, init_qmf=None):

        self.molecule = molecule
        self.n_spinorbitals = self.molecule.n_active_sos

        if self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")

        self.n_orbitals = self.n_spinorbitals // 2
        self.n_electrons = self.molecule.n_active_electrons
        self.spin = molecule.spin

        self.mapping = mapping
        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)
        self.up_then_down = up_then_down
        self.init_qmf = {"init_params": "hf-state"} if init_qmf is None else init_qmf

        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "pi_over_two", "pis", "random", "hf-state"}

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}

        # Get the fermionic Hamiltonian
        self.fermi_ham = self.molecule.fermionic_hamiltonian

        # Check if the mean-field Hamiltonian will be penalized
        if isinstance(self.init_qmf, dict):
            if "init_params" not in self.init_qmf.keys():
                err_msg = f"Missing key 'init_params' in {self.init_qmf}. "\
                          f"Supported values are {self.supported_initial_var_params}."
                raise KeyError(err_msg)
            if self.init_qmf["init_params"] in self.supported_initial_var_params:
                # Set default value of var_params and remove it
                self.var_params_default = self.init_qmf.pop("init_params")
                # Check for at least one penalty term
                if self.init_qmf:
                    # Set default values if a key has value of None
                    spin_z = self.spin // 2
                    default_keys = list(filter(lambda kv: kv[1] is None,\
                        (kv for kv in self.init_qmf.keys())))
                    for default_key in default_keys:
                        if default_key == "N":
                            self.init_qmf["N"] = (1.5, self.n_electrons)
                        elif default_key == "S^2":
                            self.init_qmf["S^2"] = (1.5, spin_z * (spin_z + 1))
                        elif default_key == "Sz":
                            self.init_qmf["Sz"] = (1.5, spin_z)
                    self.fermi_ham += penalize_mf_ham(self.init_qmf, self.n_orbitals)
                else:
                    if self.var_params_default != "hf-state":
                        warn_msg = "It is recommended to penalize the mean-field Hamiltonian with "\
                                   "one or more penalty terms is recommended when QMF params are "\
                                   "not initialized using a reference HF state."
                        warnings.warn(warn_msg, RuntimeWarning)
            else:
                err_msg = f"Unrecognized value for 'init_params' key in {self.init_qmf} "\
                          f"Supported values are {self.supported_initial_var_params}."
                raise ValueError(err_msg)
        else:
            raise TypeError(f"{self.init_qmf} must be dictionary type.")

        self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping,\
            n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons,\
            up_then_down=self.up_then_down, spin=self.spin)

        # Default starting parameters for initialization
        self.n_var_params = 2 * self.n_qubits
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
            elif var_params == "pi_over_two":
                initial_var_params = 0.5 * np.pi * np.ones((self.n_var_params,))
            elif var_params == "pis":
                initial_var_params = np.pi * np.ones((self.n_var_params,))
            elif var_params == "random":
                initial_thetas = np.pi * np.random.random((self.n_qubits,))
                initial_phis = 2. * np.pi * np.random.random((self.n_qubits,))
                initial_var_params = np.concatenate((initial_thetas, initial_phis))
            elif var_params == "hf-state":
                initial_var_params = init_qmf_state_from_hf_vec(self.n_spinorbitals,\
                    self.n_electrons, self.mapping, up_then_down=self.up_then_down, spin=self.spin)
        elif np.array(var_params).size == self.n_var_params:
            initial_var_params = np.array(var_params)
        elif np.array(var_params).size != self.n_var_params:
            err_msg = f"Expected {self.n_var_params} variational parameters but "\
                      f"received {len(var_params)}."
            raise ValueError(err_msg)

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
