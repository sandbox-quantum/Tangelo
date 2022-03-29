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

""" This module defines a class that can be used to generate the circuit that
returns or uncomputes a given statevector (takes the given statevector to the zero state).
"""

import numpy as np
import math

from tangelo.linq import Circuit, Gate


class StateVector():
    """This class provides functions to either compute a statevector (of 2**n_qubits) from the zero state
    or take that state to the zero state

    Args:
        coefficients: The list or array of coefficients defining a state. Must have length 2**n_qubits where n_qubits is an integer.
    """

    def __init__(self, coefficients, order="msq_first"):
        n_qubits = math.log2(len(coefficients))
        if n_qubits == 0 or not n_qubits.is_integer():
            raise ValueError("Length of input state must be a power of 2")
        if order not in ["msq_first", "lsq_first"]:
            raise ValueError(f"order must be 'lsq_first' or 'msq_first'")

        self.n_qubits = int(n_qubits)
        self.coeffs = coefficients
        self.order = order

    def initializing_circuit(self, return_phase=False):
        """Calculate a circuit that implements the initialization of a given statevector from the zero state
        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5
        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.

        Args:
            return_phase (bool): Return the global phase that is not captured by the circuit

        Returns:
            Circuit: The circuit that generates the statevector defined in coeffs
            float: If return_phase=True, the global phase angle not captured by the Circuit
        """
        # call to generate the circuit that takes the desired vector to zero
        disentangling_circuit, global_phase = self.uncomputing_circuit(return_phase=True)

        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        state_prep_circuit = disentangling_circuit.inverse()
        global_phase = -global_phase

        return_value = (state_prep_circuit, global_phase) if return_phase else state_prep_circuit
        return return_value

    def uncomputing_circuit(self, return_phase=False):
        """Generate a circuit that takes the desired state to the zero state.

        Args:
            return_phase (bool): Flag to return global_phase

        Returns:
            Circuit: circuit to take self.coeffs vector to :math:`|{00\\ldots0}\\rangle`
            float: (if return_phase=True) The angle that defines the global phase not captured by the circuit
        """

        circuit = Circuit(n_qubits=self.n_qubits)

        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = self.coeffs

        for i in range(self.n_qubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            remaining_param, thetas, phis = StateVector._rotations_to_disentangle(remaining_param)

            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)

            add_last_cnot = True
            if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                add_last_cnot = False

            if np.linalg.norm(phis) != 0:
                rz_mult = self._get_multiplex_circuit("RZ", phis, last_cnot=add_last_cnot)
                rz_mult.reindex_qubits(list(range(i, self.n_qubits)))
                circuit += rz_mult

            if np.linalg.norm(thetas) != 0:
                ry_mult = self._get_multiplex_circuit("RY", thetas, last_cnot=add_last_cnot)
                ry_mult.reindex_qubits(list(range(i, self.n_qubits)))
                for gate in reversed(ry_mult._gates):
                    circuit.add_gate(gate)
        global_phase = -np.angle(sum(remaining_param))

        if self.order == "lsq_first":
            circuit.reindex_qubits(list(reversed(range(0, self.n_qubits))))

        return_value = (circuit, global_phase) if return_phase else circuit
        return return_value

    @staticmethod
    def _rotations_to_disentangle(local_param):
        """Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.
        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
         [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
          0         0           Ry(theta_2^n).Rz(phi_2^n)]]

        Args:
            local_param (array): The parameters of subset of qubits to return the LSB to the zero state.

        Returns:
            list of float: remaining vector with LSB set to |0>
            list of float: The necessary RY Gate parameters
            list of float: The necessary RZ Gate parameters
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains, add_theta, add_phi) = StateVector._bloch_angles(
                local_param[2*i], local_param[2*i+1]
            )

            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(a_complex, b_complex):
        """Static internal method to work out rotation to create the passed-in
        qubit from the zero vector.

        Args:
            a_complex (complex): First complex number to calculate rotation from zero vector
            b_complex (complex): Second complex number to calculate rotation from zero vector

        Returns:
            complex: remaining phase angle not captured by RY and RZ
            float: calculated RY rotation angle
            float: calculated RZ rotation angle
        """

        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = np.absolute(a_complex)
        final_r = float(np.sqrt(mag_a**2 + np.absolute(b_complex) ** 2))
        if final_r < np.finfo(float).eps:
            theta, phi, final_r, final_t = 0, 0, 0, 0
        else:
            theta = float(2 * np.arccos(mag_a / final_r))
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.0j * final_t / 2), theta, phi

    def _get_multiplex_circuit(self, target_gate, angles, last_cnot=True) -> Circuit:
        """Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.
        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            angles (list[float]): list of rotation angles to apply Ry and Rz
            last_cnot (bool): add the last cnot if last_cnot = True

        Returns:
            Circuit: the circuit implementing the multiplexor's action
        """
        list_len = len(angles)
        local_n_qubits = int(math.log2(list_len)) + 1

        # case of no multiplexing: base case for recursion
        if local_n_qubits == 1:
            return Circuit([Gate(target_gate, 0, parameter=angles[0])], n_qubits=local_n_qubits)

        circuit = Circuit(n_qubits=local_n_qubits)

        lsb = 0
        msb = local_n_qubits - 1

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_n_qubits - 2)))

        # calc the combo angles
        angles = angle_weight.dot(np.array(angles)).tolist()

        # recursive step on half the angles fulfilling the above assumption
        multiplex_1 = self._get_multiplex_circuit(target_gate, angles[0: (list_len // 2)], False)
        circuit += multiplex_1

        # attach CNOT as follows, thereby flipping the LSB qubit
        circuit.add_gate(Gate('CNOT', target=lsb, control=msb))

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        multiplex_2 = self._get_multiplex_circuit(target_gate, angles[(list_len // 2):], False)
        if list_len > 1:
            for gate in reversed(multiplex_2._gates):
                circuit.add_gate(gate)

        else:
            circuit += multiplex_2

        # attach a final CNOT
        if last_cnot:
            circuit.add_gate(Gate("CNOT", lsb, msb))

        return circuit
