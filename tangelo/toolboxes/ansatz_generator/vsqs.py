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

from copy import deepcopy

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
    a reference_state Circuit and a hini QubitOperator.

    Args:
        molecule (SecondQuantizedMolecule): The molecular system. Default: None
        mapping (str): One of the support fermion to qubit mappings. Default : "JW"
        up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down.
            Default: False (alternating spin up/down ordering)
        intervals (int): The number of steps in the VSQS process. Must be greater than 1. Default: 2
        time (float): The propagation time. Default: 1
        qubit_hamiltonian (QubitOperator): The qubit hamiltonian to evolve. Default: None
        reference_state (Circuit): The reference state for the propagation as defined by a Circuit. Mandatory if supplying
            a qubit_hamiltonian. Default: None
        hini (QubitOperator): The initial qubit Hamiltonian that one corresponds to the reference state. Mandatory if supplying
            a qubit_hamiltonian. Default: None
        hnav (QubitOperator): The navigator Hamiltonian. Default: None
    """

    def __init__(self, molecule=None, mapping="jw", up_then_down=False, intervals=2, time=1, qubit_hamiltonian=None,
                 reference_state=None, hini=None, hnav=None, trotter_order=1):

        self.up_then_down = up_then_down
        self.mapping = mapping
        if intervals > 1:
            self.intervals = intervals
        else:
            raise ValueError("The number of intervals must be greater than 1.")
        self.time = time
        self.dt = self.time/self.intervals
        self.reference_state = reference_state
        self.trotter_order = trotter_order

        if molecule is None:
            self.qubit_hamiltonian = qubit_hamiltonian
            if not isinstance(hini, ofQubitOperator):
                raise ValueError("When providing a qubit hamiltonian, an initial Hamiltonian must also be provided")
            self.hini = hini
            if not isinstance(reference_state, Circuit):
                raise ValueError("Reference state must be provided when simulating a qubit hamiltonian directly")
            self.reference_state = reference_state
        else:
            self.n_electrons = molecule.n_active_electrons
            self.n_spinorbitals = int(molecule.n_sos)
            self.n_qubits = get_qubit_number(mapping, self.n_spinorbitals)
            self.spin = molecule.spin
            self.qubit_hamiltonian = fermion_to_qubit_mapping(molecule.fermionic_hamiltonian, n_spinorbitals=self.n_spinorbitals,
                                                              n_electrons=self.n_electrons, mapping=self.mapping,
                                                              up_then_down=self.up_then_down, spin=self.spin)
            self.hini = self.build_hini(molecule) if hini is None else hini
        self.hfin = self.qubit_hamiltonian
        self.hnav = hnav

        self.hfin_list, self.n_hfin = self.remove_constant_and_count_terms(self.qubit_hamiltonian)
        self.hini_list, self.n_hini = self.remove_constant_and_count_terms(self.hini)
        if self.hnav is None:
            self.stride = 2
            self.n_hnav = 0
        else:
            if isinstance(self.hnav, ofQubitOperator):
                self.stride = 3
                self.hnav_list, self.n_hnav = self.remove_constant_and_count_terms(self.hnav)
            else:
                raise ValueError("Navigator Hamiltonian must be a QubitOperator")

        self.n_var_params = (intervals - 1) * self.stride
        self.n_var_gates = (self.n_hini + self.n_hfin + self.n_hnav) * self.trotter_order

        self.var_params = None
        self.circuit = None

    def remove_constant_and_count_terms(self, qu_op):
        """count of non-zero terms in a QubitOperator and return the list of terms.items()"""
        count = 0
        new_qu_op = deepcopy(qu_op)
        for term in new_qu_op.terms.keys():
            if term:
                count += 1
            else:
                new_qu_op.terms[term] = 0
        new_qu_op.compress()
        return list(new_qu_op.terms.items()), count

    def build_hini(self, molecule):
        """Return the initial Hamiltonian (hini) composed of the one-body terms derived from the diagonal of Fock
        matrix and one-body off-diagonal terms"""
        fock = molecule.mean_field.mo_coeff.T @ molecule.mean_field.get_fock() @ molecule.mean_field.mo_coeff
        h1 = molecule.mean_field.mo_coeff.T @ molecule.mean_field.get_hcore() @ molecule.mean_field.mo_coeff
        hf_ferm = FermionOperator()
        for i in range(self.n_spinorbitals//2):
            for j in range(self.n_spinorbitals//2):
                if i != j:
                    hf_ferm += FermionOperator(((i * 2, 1), (j * 2, 0)), h1[i, j])
                    hf_ferm += FermionOperator(((i * 2 + 1, 1), (j * 2 + 1, 0)), h1[i, j])
                else:
                    hf_ferm += FermionOperator(((i * 2, 1), (j * 2, 0)), fock[i, j])
                    hf_ferm += FermionOperator(((i * 2 + 1, 1), (j * 2 + 1, 0)), fock[i, j])
        return fermion_to_qubit_mapping(hf_ferm, mapping=self.mapping, n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons,
                                        up_then_down=self.up_then_down, spin=self.spin)

    def set_var_params(self, var_params=None):
        """Set values for the variational parameters. Default is linear interpolation."""
        if var_params is None:
            var_params = self.init_params()[self.stride:self.n_var_params+self.stride]

        init_var_params = np.array(var_params)
        if init_var_params.size == self.n_var_params:
            self.var_params = var_params
        else:
            raise ValueError(f"Expected {self.n_var_params} variational parameters but received {init_var_params.size}.")
        return var_params

    def update_var_params(self, var_params):
        """Update the variational parameters in the circuit without rebuilding."""
        for i in range(self.intervals-1):
            self._update_gate_params_for_qu_op(self.hini_list, self.n_var_gates * i, var_params[self.stride*i], self.n_hini)
            self._update_gate_params_for_qu_op(self.hfin_list, self.n_var_gates * i + self.n_hini * self.trotter_order,
                                               var_params[self.stride*i+1], self.n_hfin)
            if self.hnav is not None:
                self._update_gate_params_for_qu_op(self.hnav_list, self.n_var_gates * i + (self.n_hini + self.n_hfin) * self.trotter_order,
                                                   var_params[self.stride*i+2], self.n_hnav)

    def _update_gate_params_for_qu_op(self, qu_op, n_var_start, var_param, num_terms):
        prefac = 2 / self.trotter_order * self.dt * var_param
        for i, (_, coeff) in enumerate(qu_op):
            self.circuit._variational_gates[n_var_start+i].parameter = prefac * coeff
        if self.trotter_order == 2:
            for i, (_, coeff) in enumerate(list(reversed(qu_op))):
                self.circuit._variational_gates[n_var_start+num_terms+i].parameter = prefac * coeff

    def init_params(self):
        """Generate initial parameters for the VSQS algorithm.
        a_i = step*i, b_i=1-step*i, c_i= 1-step*i i<n_intervals/2, step*i i>n_intervals/2"""
        a = np.zeros(self.intervals+1)
        b = np.zeros(self.intervals+1)
        c = np.zeros(self.intervals+1)
        a[0] = 1
        b[self.intervals] = 1
        step = 1/self.intervals
        for i in range(1, self.intervals):
            a[i] = (1 - i * step)
            b[i] = (i * step)
        all_params = np.zeros(self.stride * (self.intervals + 1))
        if self.hnav is None:
            for i in range(self.intervals + 1):
                all_params[2 * i] = a[i]
                all_params[2 * i + 1] = b[i]
        else:
            c[0:self.intervals//2] = b[0:self.intervals//2]
            c[self.intervals//2:self.intervals+1] = a[self.intervals//2:self.intervals+1]
            for i in range(self.intervals + 1):
                all_params[3 * i] = a[i]
                all_params[3 * i + 1] = b[i]
                all_params[3 * i + 2] = c[i]
        return all_params

    def prepare_reference_state(self):
        """Prepare a circuit generating the HF reference state."""
        return get_reference_circuit(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.mapping,
                                     up_then_down=self.up_then_down, spin=self.spin)

    def build_circuit(self, var_params=None):
        """Build the VSQS circuit by successive first-order trotterizations of hini, hfin and possibly hnav"""
        reference_state_circuit = self.prepare_reference_state() if self.reference_state is None else self.reference_state
        self.var_params = self.set_var_params(var_params)

        vsqs_circuit = get_exponentiated_qubit_operator_circuit(self.hini, time=self.dt, trotter_order=self.trotter_order, pauli_order=self.hini_list)
        for i in range(self.intervals-1):
            vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.hini, time=self.var_params[i * self.stride] * self.dt, variational=True,
                                                                     trotter_order=self.trotter_order, pauli_order=self.hini_list)
            vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.hfin, time=self.var_params[i * self.stride + 1] * self.dt, variational=True,
                                                                     trotter_order=self.trotter_order, pauli_order=self.hfin_list)
            if self.hnav is not None:
                vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.hnav, time=self.var_params[i * self.stride + 2] * self.dt, variational=True,
                                                                         trotter_order=self.trotter_order, pauli_order=self.hnav_list)
        vsqs_circuit += get_exponentiated_qubit_operator_circuit(self.hfin, time=self.dt, trotter_order=self.trotter_order, pauli_order=self.hfin_list)

        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + vsqs_circuit
        else:
            self.circuit = vsqs_circuit

        return self.circuit
