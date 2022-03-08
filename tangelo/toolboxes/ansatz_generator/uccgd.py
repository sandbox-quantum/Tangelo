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

"""This module defines the UCCGD ansatz class. It provides a chemically
inspired ansatz and is an implementation of the classical unitary CCGD ansatz.
Generalized double excitation determinants, in accordance
with the system number of electron and spin, are considered. For more
information about this ansatz, see references below.

Refs:

"""

import numpy as np
from openfermion import hermitian_conjugated
from openfermion import FermionOperator as ofFermionOperator

from tangelo.linq import Circuit
from tangelo.toolboxes.operators.operators import FermionOperator

from .ansatz import Ansatz
from .ansatz_utils import exp_pauliword_to_gates, trotterize
from ._unitary_cc_paired import get_upccgsd
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit


class UCCGD(Ansatz):
    """This class implements the UpCCGSD ansatz. This implies that the
    mean-field is computed with the RHF or ROHF reference integrals.

    Args:
        molecule (SecondQuantizedMolecule) : The molecular system.
        mapping (str) : one of the supported qubit mapping identifiers. Default:
            "JW".
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False (i.e. has
            alternating spin up/down ordering).
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, k=2):

        self.n_spinorbitals = molecule.n_active_sos
        self.n_electrons = molecule.n_active_electrons
        self.spin = molecule.spin
        self.k = k

        self.qubit_mapping = mapping
        self.up_then_down = up_then_down

        # Later: refactor to handle various flavors of UCCSD
        if self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even.")
        self.n_spatial_orbitals = self.n_spinorbitals // 2
        n_mos = self.n_spatial_orbitals
        p = 0
        for u in range(n_mos):
            for w in range(u, n_mos):
                for v in range(w, n_mos):
                    for t in range(v, n_mos):
                        if len(set([u, w, v, t])) >= 1:  # are they not all equal
                            p = p + 1
        self.n_var_params = p

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF", "None"}
        # Supported var param initialization
        self.var_params_default = "random"
        self.supported_initial_var_params = {"ones", "random"}

        # Default initial parameters for initialization
        self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros, random numbers,
        providing some keywords for users, and also supporting direct user input
        (list or numpy array). Return the parameters so that workflows such as
        VQE can retrieve these values.
        """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = 1.e-1 * (np.random.random((self.n_var_params,)) - 0.5)
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g
        prepare reference wavefunction with HF, multi-reference state, etc).
        These preparations must be consistent with the transform used to obtain
        the qubit operator.
        """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.default_reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                         n_electrons=self.n_electrons,
                                         mapping=self.qubit_mapping,
                                         up_then_down=self.up_then_down,
                                         spin=self.spin)
        if self.default_reference_state == "None":
            return Circuit()

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state
        preparation ansatz (with currently specified initial_state and
        var_params).
        """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Prepare reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Build qubit operator required to build k-UpCCGSD

        # Build dictionary of ordered pauli words for fast update of parameters intead of rebuilding circuit
        # Initialize dictionary of, dictionaries for each UpCCGSD step (current_k)
        self.pauli_to_angles_mapping = dict()

        qubit_op = self._get_qubit_operator()

        # Initialize dictionary of qubit_op terms for each UpCCGSD step
        self.pauli_to_angles_mapping = dict()

        upccgsd_circuit = trotterize(qubit_op, variational=True)

        # skip over the reference state circuit if it is empty
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + upccgsd_circuit
        else:
            self.circuit = upccgsd_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the already-built
        ansatz circuit member. Preferable to rebuilt your circuit from scratch,
        which can be an involved process.
        """

        self.set_var_params(var_params)

        # Loop through each current_k step
        self.build_circuit(var_params)

    def _get_qubit_operator(self):
        """Construct UpCCGSD FermionOperator for variational
        parameters, and translate to QubitOperator via relevant qubit mapping.

        Returns:
            QubitOperator: qubit-encoded elements of the UCCGD
        """
        fermion_op = ofFermionOperator()
        n_mos = self.n_spinorbitals // 2
        p = -1
        for u in range(n_mos):
            for w in range(u, n_mos):
                for v in range(w, n_mos):
                    for t in range(v, n_mos):
                        if len(set([u, w, v, t])) >= 1:  # are they not all equal
                            p = p + 1
                            for sig in range(2):
                                for tau in range(2):
                                    c_op = ofFermionOperator(((2*t+sig, 1), (2*v+tau, 1), (2*w+tau, 0), (2*u+sig, 0)), self.var_params[p])
                                    fermion_op += c_op - hermitian_conjugated(c_op)
                            for sig in range(2):
                                for tau in range(2):
                                    c_op = ofFermionOperator(((2*v+sig, 1), (2*t+tau, 1), (2*u+tau, 0), (2*w+sig, 0)), self.var_params[p])
                                    fermion_op += c_op - hermitian_conjugated(c_op)

        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.qubit_mapping,
                                            n_spinorbitals=self.n_spinorbitals,
                                            n_electrons=self.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        # qubit_op.compress()
        return qubit_op
