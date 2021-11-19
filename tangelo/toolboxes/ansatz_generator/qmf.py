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

from .ansatz import Ansatz
from ._qubit_mf import get_qmf_circuit, initialize_qmf_state_from_hf_vec
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number, fermion_to_qubit_mapping 

class QMF(Ansatz):
    """
        This class implements the qubit mean field (QMF) ansatz. Currently, only closed-shell QMF is supported.
        This implies that the mean-field is computed with the RHF reference integrals.

        Args:
            molecule (SecondQuantizedMolecule) : The molecular system.
            mapping (str) : see mapping_transform.py for options 'JW' (Jordan Wigner), or 'BK' (Bravyi Kitaev), or 'SCBK' (symmetry-conserving Bravyi Kitaev)
                Default, 'JW'
            up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down
                Default, False (i.e. has alternating spin up/down ordering).
    """
 
    def __init__(self, molecule, mapping="JW", up_then_down=False):
     
        self.molecule = molecule
        self.mapping = mapping
        self.up_then_down = up_then_down

        if self.mapping.upper() == 'JW':
            if not self.up_then_down:
                warnings.warn("When using the QMF and QCC ansatz classes together, instantiate both classes with up_then_down = True if mapping = 'JW'.\n", RuntimeWarning)

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

        self.n_var_params = 2 * self.n_qubits

        # Supported reference state initialization
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "random", "hf-state"}
        self.var_params = None

        # Default starting parameters for initialization
        self.default_reference_state = "HF"
        self.var_params_default = "hf-state"
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, Hartree-Fock state, providing some
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
                initial_var_params = np.pi * np.random.random((self.n_qubts,))
                initial_var_params = np.concatenate((initial_var_params, 2. * np.pi * np.random.random((self.n_qubits,))))
            elif var_params == "hf-state":
                initial_var_params = initialize_qmf_state_from_hf_vec(self.n_spinorbitals, self.n_electrons, self.mapping, self.up_then_down) 
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
            return get_qmf_circuit(self.var_params)

    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build the circuit for the QMF ansatz
        self.circuit = self.prepare_reference_state()

    def update_var_params(self, var_params):
        """ 
        Update Rebuild the QMF circuit with the current state of var_params 
        """
 
        # The QMF operators do not change -- update parameters and rebuild the circuit
        self.set_var_params(var_params)
        self.build_circuit(var_params)

