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

"""This module defines the Variationally Scheduled Quantum Simulation class. It provides an
Adiabatic State Preparation (ASP) inspired ansatz as described in https://arxiv.org/abs/2003.09913."""

import numpy as np
from openfermion import QubitOperator as ofQubitOperator

from .ansatz import Ansatz
from .ansatz_utils import get_exponentiated_qubit_operator_circuit
from tangelo.toolboxes.operators import FermionOperator
from tangelo.linq import Circuit
from tangelo.toolboxes.qubit_mappings.mapping_transform import get_qubit_number, fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit


class VSQS(Ansatz):
    """This class implements the Variationally Scheduled Quantum Simulator (VSQS) for state preparation as described in
    https://arxiv.org/abs/2003.09913

    Must supply either a molecule or a qubit_hamiltonian. If supplying a qubit_hamiltonian, must also supply
    a reference_state Circuit and a h_init QubitOperator.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system. Default: None
        mapping (str): One of the supported fermion to qubit mappings. Default : "JW"
        up_then_down (bool): Change basis ordering, putting all spin up orbitals first, followed by all spin down.
            Default: False (alternating spin up/down ordering)
        intervals (int): The number of steps in the VSQS process. Must be greater than 1. Default: 2
        time (float): The propagation time. Default: 1.
        qubit_hamiltonian (QubitOperator): The qubit Hamiltonian to evolve. Default: None
        reference_state (Circuit): The reference state for the propagation as defined by a Circuit. Mandatory if supplying
            a qubit_hamiltonian. Default: None
        h_init (QubitOperator): The initial qubit Hamiltonian that corresponds to the reference state. Mandatory if supplying
            a qubit_hamiltonian. Default: None
        h_nav (QubitOperator): The navigator Hamiltonian. Default: None
        trotter_order (int): The order of the Trotterization for each qubit operator. Default: 1
    """

    def __init__(self, molecule=None, mapping="jw", up_then_down=False, intervals=2, time=1., qubit_hamiltonian=None,
                 reference_state=None, h_init=None, h_nav=None, trotter_order=1):

        self.up_then_down = up_then_down
        self.mapping = mapping
        if intervals > 1:
            self.intervals = intervals
        else:
            raise ValueError("The number of intervals must be greater than 1.")
        self.time = time
        self.dt = self.time/self.intervals
        self.reference_state = reference_state
        if trotter_order in [1, 2]:
            self.trotter_order = trotter_order
        else:
            raise ValueError("Only trotter_order = 1, 2 is supported")

        if molecule is None:
            self.qubit_hamiltonian = qubit_hamiltonian
            if not isinstance(h_init, ofQubitOperator):
                raise ValueError("When providing a qubit hamiltonian, an initial qubit Hamiltonian must also be provided")
            self.h_init = h_init
            if not isinstance(reference_state, Circuit):
                raise ValueError("Reference state Circuit must be provided when simulating a qubit hamiltonian directly")
            self.reference_state = reference_state
        else:
            self.n_electrons = molecule.n_active_electrons
            self.n_spinorbitals = int(molecule.n_sos)
            self.n_qubits = get_qubit_number(mapping, self.n_spinorbitals)
            self.spin = molecule.spin
            self.qubit_hamiltonian = fermion_to_qubit_mapping(molecule.fermionic_hamiltonian, n_spinorbitals=self.n_spinorbitals,
                                                              n_electrons=self.n_electrons, mapping=self.mapping,
                                                              up_then_down=self.up_then_down, spin=self.spin)
            self.h_init = self._build_h_init(molecule) if h_init is None else h_init
        self.h_final = self.qubit_hamiltonian
        self.h_nav = h_nav

        def qu_op_to_list(qu_op):
            """Remove consant term and convert QubitOperator to list of (term, coeff)"""
            new_qu_op = qu_op - qu_op.constant
            new_qu_op.compress()
            return list(new_qu_op.terms.items())

        self.h_final_list = qu_op_to_list(self.h_final)
        self.n_h_final = len(self.h_final_list)
        self.h_init_list = qu_op_to_list(self.h_init)
        self.n_h_init = len(self.h_init_list)
        if self.h_nav is None:
            self.stride = 2
            self.n_h_nav = 0
        else:
            if isinstance(self.h_nav, ofQubitOperator):
                self.stride = 3
                self.h_nav_list = qu_op_to_list(self.h_nav)
                self.n_h_nav = len(self.h_nav_list)
            else:
                raise ValueError("Navigator Hamiltonian must be a QubitOperator")

        self.n_var_params = (intervals - 1) * self.stride
        self.n_var_gates = (self.n_h_init + self.n_h_final + self.n_h_nav) * self.trotter_order

        self.var_params = None
        self.circuit = None

    def _build_h_init(self, molecule):
        """Return the initial Hamiltonian (h_init) composed of the one-body terms derived from the diagonal of Fock
        matrix and one-body off-diagonal terms"""
        core_constant, h1, two_body = molecule.get_active_space_integrals()
        diag_fock = np.diag(h1).copy()
        n_active_occupied = len(molecule.active_occupied)
        for j in range(molecule.n_active_mos):
            for i in range(n_active_occupied):
                diag_fock[j] += 2*two_body[i, j, j, i] - 1*two_body[i, j, i, j]

        hf_ferm = FermionOperator((), core_constant)
        for i in range(molecule.n_active_mos):
            for j in range(molecule.n_active_mos):
                if i != j:
                    hf_ferm += FermionOperator(((i * 2, 1), (j * 2, 0)), h1[i, j])
                    hf_ferm += FermionOperator(((i * 2 + 1, 1), (j * 2 + 1, 0)), h1[i, j])
                else:
                    hf_ferm += FermionOperator(((i * 2, 1), (j * 2, 0)), diag_fock[i])
                    hf_ferm += FermionOperator(((i * 2 + 1, 1), (j * 2 + 1, 0)), diag_fock[j])
        return fermion_to_qubit_mapping(hf_ferm, mapping=self.mapping, n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons,
                                        up_then_down=self.up_then_down, spin=self.spin)

    def set_var_params(self, var_params=None):
        """Set values for the variational parameters. Default is linear interpolation."""
        if var_params is None:
            var_params = self._init_params()[self.stride:self.n_var_params+self.stride]

        init_var_params = np.array(var_params)
        if init_var_params.size == self.n_var_params:
            self.var_params = var_params
        else:
            raise ValueError(f"Expected {self.n_var_params} variational parameters but received {init_var_params.size}.")
        return var_params

    def update_var_params(self, var_params):
        """Update the variational parameters in the circuit without rebuilding."""
        for i in range(self.intervals-1):
            self._update_gate_params_for_qu_op(self.h_init_list, self.n_var_gates * i, var_params[self.stride*i], self.n_h_init)
            self._update_gate_params_for_qu_op(self.h_final_list, self.n_var_gates * i + self.n_h_init * self.trotter_order,
                                               var_params[self.stride*i+1], self.n_h_final)
            if self.h_nav is not None:
                self._update_gate_params_for_qu_op(self.h_nav_list, self.n_var_gates * i + (self.n_h_init + self.n_h_final) * self.trotter_order,
                                                   var_params[self.stride*i+2], self.n_h_nav)

    def _update_gate_params_for_qu_op(self, qu_op_list, n_var_start, var_param, num_terms):
        """Updates the corresponding circuit variational_gates for a QubitOperator defined by term order qu_op_list

        Args:
            qu_op_list :: The list with elements (term, coeff) that defines the trotterization of a QubitOperator
            n_var_start :: The varational_gates position that the trotterization of qu_op starts
            var_param :: The variational parameter (evolution time) for qu_op. Same for all terms in qu_op
            num_terms :: The number of terms in qu_op
        """
        prefac = 2 / self.trotter_order * self.dt * var_param
        for i, (_, coeff) in enumerate(qu_op_list):
            self.circuit._variational_gates[n_var_start+i].parameter = prefac * coeff
        if self.trotter_order == 2:
            for i, (_, coeff) in enumerate(list(reversed(qu_op_list))):
                self.circuit._variational_gates[n_var_start+num_terms+i].parameter = prefac * coeff

    def _init_params(self):
        """Generate initial parameters for the VSQS algorithm.
        a_i = step*i, b_i=1-step*i, c_i= 1-step*i i<=n_intervals/2, step*i i>n_intervals/2
        """
        a = np.zeros(self.intervals+1)
        b = np.zeros(self.intervals+1)
        a[0] = 1
        b[self.intervals] = 1
        step_size = 1/self.intervals
        for i in range(1, self.intervals):
            a[i] = (1 - i * step_size)
            b[i] = (i * step_size)
        all_params = np.zeros(self.stride * (self.intervals + 1))
        if self.h_nav is None:
            # order [a[0], b[0], a[1], b[1],...]
            all_params = np.dstack((a, b)).flatten()
        else:
            c = np.zeros(self.intervals+1)
            c[0:self.intervals//2] = b[0:self.intervals//2]
            c[self.intervals//2:self.intervals+1] = a[self.intervals//2:self.intervals+1]
            # order [a[0], b[0], c[0], a[1], b[1], c[1],...]
            all_params = np.dstack((a, b, c)).flatten()
        return all_params

    def prepare_reference_state(self):
        """Prepare a circuit generating the HF reference state."""
        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping,
                                     up_then_down=self.up_then_down, spin=self.spin)

    def build_circuit(self, var_params=None):
        """Build the VSQS circuit by successive first- or second-order trotterizations of h_init, h_final and possibly h_nav"""
        reference_state_circuit = self.prepare_reference_state() if self.reference_state is None else self.reference_state
        self.var_params = self.set_var_params(var_params)

        vsqs_circuit = get_exponentiated_qubit_operator_circuit(self.h_init, time=self.dt, trotter_order=self.trotter_order, pauli_order=self.h_init_list)
        for i in range(self.intervals-1):
            vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.h_init, time=self.var_params[i * self.stride] * self.dt, variational=True,
                                                                     trotter_order=self.trotter_order, pauli_order=self.h_init_list)
            vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.h_final, time=self.var_params[i * self.stride + 1] * self.dt, variational=True,
                                                                     trotter_order=self.trotter_order, pauli_order=self.h_final_list)
            if self.h_nav is not None:
                vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.h_nav, time=self.var_params[i * self.stride + 2] * self.dt, variational=True,
                                                                         trotter_order=self.trotter_order, pauli_order=self.h_nav_list)
        vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.h_final, time=self.dt, trotter_order=self.trotter_order, pauli_order=self.h_final_list)

        self.circuit = reference_state_circuit + vsqs_circuit if reference_state_circuit.size != 0 else vsqs_circuit

        return self.circuit
