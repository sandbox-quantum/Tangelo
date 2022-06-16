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
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.ansatz_generator._qubit_mf import get_qmf_circuit, init_qmf_from_hf,\
                                                         penalize_mf_ham, init_qmf_from_vector


class QMF(Ansatz):
    """This class implements the QMF ansatz. Closed-shell and restricted open-shell QMF are
    supported. While the form of the QMF ansatz is the same for either variation, the underlying
    fermionic mean-field state is treated differently depending on the spin. Closed-shell
    or restricted open-shell QMF implies that spin = 0 or spin != 0 and the fermionic mean-field
    state is obtained using a RHF or ROHF Hamiltonian, respectively.

    Optimizing QMF variational parameters can be risky without taking proper precautions,
    especially when a random initial guess is used. It is recommended that penalty terms are
    added to the mean-field Hamiltonian to enforce appropriate electron number and spin angular
    momentum symmetries on the QMF wave function during optimization (see Ref. 4). If using
    penalty terms is to be avoided, an initial guess based on a Hartree-Fock reference state will
    likely converge quickly to the desired state, but this is not guaranteed.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False.
        init_qmf (dict): Controls for QMF variational parameter initialization and mean-field
            Hamiltonian penalization. Supported keys are "init_params", "N", "S^2", or "Sz" (str).
            Values of "init_params" must be in self.supported_initial_var_params (str). Values of
            "N", "S^2", or "Sz" are (tuple or None). If tuple, the elements are a penalty
            term coefficient, mu (float), and a target value of the penalty operator (int).
            Example - "key": (mu, target). If "N", "S^2", or "Sz" is None, a penalty term is added
            with default mu and target values: mu = 1.5 and target is derived from molecule as
            <N> = n_electrons, <S^2> = spin_z * (spin_z + 1), and <Sz> = spin_z, where
            spin_z = spin // 2. Key, value pairs are case sensitive and mu > 0.
            Default, {"init_params": "hf_state"}.
    """

    def __init__(self, molecule, mapping="jw", up_then_down=False, init_qmf=None):

        if not molecule:
            raise ValueError("An instance of SecondQuantizedMolecule is required for initializing "
                             "the self.__class__.__name__ ansatz class.")
        self.molecule = molecule

        self.n_spinorbitals = self.molecule.n_active_sos
        if self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")

        self.n_orbitals = self.n_spinorbitals // 2
        self.spin = molecule.spin
        self.fermi_ham = self.molecule.fermionic_hamiltonian
        self.n_electrons = self.molecule.n_active_electrons

        self.mapping = mapping
        self.n_qubits = get_qubit_number(self.mapping, self.n_spinorbitals)
        self.up_then_down = up_then_down
        self.init_qmf = {"init_params": "hf_state"} if init_qmf is None else init_qmf

        # Supported var param initialization
        self.supported_initial_var_params = {"vacuum", "half_pi", "minus_half_pi", "full_pi",
                                             "random", "hf_state", "vector"}

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}

        # Add any penalty terms to the fermionic Hamiltonian
        if isinstance(self.init_qmf, dict):
            if "init_params" not in self.init_qmf.keys():
                raise KeyError(f"Missing key 'init_params' in {self.init_qmf}. "
                               f"Supported values are {self.supported_initial_var_params}.")
            if self.init_qmf["init_params"] in self.supported_initial_var_params:
                # Set the default QMF parameter procedure
                self.var_params_default = self.init_qmf.pop("init_params")
                if self.var_params_default == "vector":
                    self.vector = self.init_qmf.pop("vector")
                # Check for at least one penalty term
                if self.init_qmf:
                    # Set default penalty term values
                    spin_z = self.spin // 2
                    init_qmf_defaults = {"N": (1.5, self.n_electrons),
                                         "S^2": (1.5, spin_z * (spin_z + 1)), "Sz": (1.5, spin_z)}
                    # Check if the user requested default values for any penalty terms
                    for term, params in init_qmf_defaults.items():
                        if params is None:
                            self.init_qmf[term] = init_qmf_defaults[term]
                    # Add the penalty terms to the mean-field Hamiltonian
                    self.fermi_ham += penalize_mf_ham(self.init_qmf, self.n_orbitals)
                else:
                    if self.var_params_default != "hf_state":
                        warnings.warn("It is recommended that the QMF parameters are initialized "
                                      "using a Hartree-Fock reference state if penalty terms are "
                                      "not added to the mean-field Hamiltonian.", RuntimeWarning)
            else:
                raise ValueError(f"Unrecognized value for 'init_params' key in {self.init_qmf} "
                                 f"Supported values are {self.supported_initial_var_params}.")
        else:
            raise TypeError(f"{self.init_qmf} must be dictionary type.")

        # Get the qubit Hamiltonian
        self.qubit_ham = fermion_to_qubit_mapping(self.fermi_ham, self.mapping, self.n_spinorbitals,
                                                  self.n_electrons, self.up_then_down, self.spin)

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
                raise ValueError(f"Supported keywords for initializing variational parameters: "
                                 f"{self.supported_initial_var_params}")
            # Initialize |QMF> as |00...0>
            if var_params == "vacuum":
                initial_var_params = np.zeros((self.n_var_params,), dtype=float)
            # Initialize |QMF> as (1/sqrt(2))^n_qubits * tensor_prod(|0> + 1j|1>)
            elif var_params == "half_pi":
                initial_var_params = 0.5 * np.pi * np.ones((self.n_var_params,))
            # Initialize |QMF> as (1/sqrt(2))^n_qubits * tensor_prod(|0> - 1j|1>)
            elif var_params == "minus_half_pi":
                initial_var_params = -0.5 * np.pi * np.ones((self.n_var_params,))
            # Initialize |QMF> as (i)^n_qubits * |11...1>
            elif var_params == "full_pi":
                initial_var_params = np.pi * np.ones((self.n_var_params,))
            # Random initialization of thetas over [0, pi] and phis over [0, 2 * pi]
            elif var_params == "random":
                initial_thetas = np.pi * np.random.random((self.n_qubits,))
                initial_phis = 2. * np.pi * np.random.random((self.n_qubits,))
                initial_var_params = np.concatenate((initial_thetas, initial_phis))
            # Initialize thetas as 0 or pi such that |QMF> = |HF> and set all phis to 0
            elif var_params == "hf_state":
                initial_var_params = init_qmf_from_hf(self.n_spinorbitals, self.n_electrons,
                                                      self.mapping, self.up_then_down, self.spin)
            elif var_params == "vector":
                initial_var_params = init_qmf_from_vector(self.vector, self.mapping, self.up_then_down)
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
                             f"{self.supported_reference_state}")
        if self.default_reference_state == "HF":
            reference_state_circuit = get_qmf_circuit(self.var_params, True)
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
