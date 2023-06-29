# Copyright 2023 Good Chemistry Company.
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

""" Base class to define backends (simulators or actual devices) and abstracting their
differences from the user. Users can define their own backend as a child class of this one.
Able to run noiseless and noisy simulations, leveraging the capabilities of different backends,
quantum or classical.

If the user provides a noise model, then a noisy simulation is run with n_shots
shots. If the user only provides n_shots, a noiseless simulation is run, drawing
the desired amount of shots. If the target backend has access to the statevector
representing the quantum state, we leverage any kind of emulation available to
reduce runtime (directly generating shot values from final statevector etc). If
the quantum circuit contains a MEASURE instruction, it is assumed to simulate a
mixed-state and the simulation will be carried by simulating individual shots
(e.g a number of shots is required).

Some backends may only support a subset of the above. This information is
contained in the static method backend_info defined by each subclass target.
"""

import abc
import math
from collections import Counter

import numpy as np
from scipy import stats
from bitarray import bitarray

from tangelo.linq import Gate, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
from tangelo.toolboxes.operators import QubitOperator


def get_expectation_value_from_frequencies_oneterm(term, frequencies):
    """Return the expectation value of a single-term qubit-operator, given
    the result of a state-preparation.

    Args:
        term (QubitOperator): a single-term qubit operator.
        frequencies (dict): histogram of frequencies of measurements (assumed
            to be in lsq-first format).

    Returns:
        complex: The expectation value of this operator with regard to the
            state preparation.
    """

    if not frequencies.keys():
        return ValueError("Must pass a non-empty dictionary of frequencies.")
    n_qubits = len(list(frequencies.keys())[0])

    # Get term mask
    mask = ["0"] * n_qubits
    for index, op in term:
        mask[index] = "1"
    mask = "".join(mask)

    # Compute expectation value of the term
    expectation_term = 0.
    for basis_state, freq in frequencies.items():
        # Compute sample value using state_binstr and term mask, update term expectation value
        sample = (-1) ** ((bitarray(mask) & bitarray(basis_state)).to01().count("1") % 2)
        expectation_term += sample * freq

    return expectation_term


def get_variance_from_frequencies_oneterm(term, frequencies):
    """Return the variance of the expectation value of a single-term qubit-operator, given
    the result of a state-preparation.
    Args:
        term (QubitOperator): a single-term qubit operator.
        frequencies (dict): histogram of frequencies of measurements (assumed
            to be in lsq-first format).
    Returns:
        complex: The variance of this operator with regard to the
            state preparation.
    """

    if not frequencies.keys():
        return ValueError("Must pass a non-empty dictionary of frequencies.")
    n_qubits = len(list(frequencies.keys())[0])

    # Get term mask
    mask = ["0"] * n_qubits
    for index, op in term:
        mask[index] = "1"
    mask = "".join(mask)

    # Compute expectation value of the term
    expectation_term = get_expectation_value_from_frequencies_oneterm(term, frequencies)
    variance_term = 0.
    for basis_state, freq in frequencies.items():
        # Compute sample variance using state_binstr and term mask, update term variance
        sample = (-1) ** ((bitarray(mask) & bitarray(basis_state)).to01().count("1") % 2)
        variance_term += freq*(expectation_term - sample)**2

    return variance_term


def collapse_statevector_to_desired_measurement(statevector, qubit, result, order="lsq_first"):
    """Take 0 or 1 part of a statevector for a given qubit and return a normalized statevector and probability.

    Args:
        statevector (array): The statevector for which the collapse to the desired qubit value is performed.
        qubit (int): Index of target qubit to collapse in the desired classical state.
        result (int): 0 or 1.
        order (string): The qubit ordering of the statevector, lsq_first or msq_first.

    Returns:
        array: The collapsed and renormalized statevector.
        float: The probability for the desired measurement to occur.
    """

    n_qubits = round(math.log2(len(statevector)))

    if 2**n_qubits != len(statevector):
        raise ValueError(f"Statevector length of {len(statevector)} is not a power of 2.")

    if qubit > n_qubits-1:
        raise ValueError("qubit index to measure is larger than number of qubits in statevector")

    if result not in {0, 1}:
        raise ValueError(f"Result is not valid, must be an integer of 0 or 1 but received {result}")

    if order.lower() not in {"lsq_first", "msq_first"}:
        raise ValueError("Order must be lsq_first or msq_first")

    before_index_length = 2**qubit if order == "lsq_first" else 2**(n_qubits-1-qubit)
    after_index_length = 2**(n_qubits-1-qubit) if order == "lsq_first" else 2**qubit

    sv_selected = np.reshape(statevector, (before_index_length, 2, after_index_length))
    sv_selected[:, (result + 1) % 2, :] = 0
    sv_selected = sv_selected.flatten()

    sqrt_probability = np.linalg.norm(sv_selected)
    if sqrt_probability < 1.e-14:
        raise ValueError(f"Probability of desired measurement={0} for qubit={qubit} is zero.")

    sv_selected = sv_selected/sqrt_probability  # casting issue if inplace for probability 1

    return sv_selected, sqrt_probability**2


class Backend(abc.ABC):

    def __init__(self, n_shots=None, noise_model=None):
        """Instantiate Backend object.

        Args:
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format
                expected from the target backend.
        """

        self._current_state = None
        self._noise_model = noise_model

        # Can be modified later by user as long as it retains the same type (ex: cannot change to/from None)
        self.n_shots = n_shots
        self.freq_threshold = 1e-10

        # Set additional attributes related to the target backend chosen by the user
        for k, v in self.backend_info().items():
            setattr(self, k, v)

        # Raise error if user attempts to pass a noise model to a backend not supporting noisy simulation
        if self._noise_model and not self.noisy_simulation:
            raise ValueError("Target backend does not support noise models.")

        # Raise error if the number of shots has not been passed for a noisy simulation or if statevector unavailable
        if not self.n_shots and (not self.statevector_available or self._noise_model):
            raise ValueError("A number of shots needs to be specified.")

    @abc.abstractmethod
    def simulate_circuit(self):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit (Circuit): a circuit in the abstract format to be translated
                for the target backend.
            return_statevector (bool): option to return the statevector as well,
                if available.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            save_mid_circuit_meas (bool): Save mid-circuit measurement results to
                self.mid_circuit_meas_freqs. All measurements will be saved to
                self.all_frequencies, with keys of length (n_meas + n_qubits).
                The leading n_meas values will hold the results of the MEASURE gates,
                ordered by their appearance in the source_circuit.
                The last n_qubits values will hold the measurements performed on
                each of qubits at the end of the circuit.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        pass

    def simulate(self, source_circuit, return_statevector=False, initial_statevector=None,
                 desired_meas_result=None, save_mid_circuit_meas=False):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit (Circuit): a circuit in the abstract format to be translated
                for the target backend.
            return_statevector (bool): option to return the statevector as well,
                if available.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The binary string of the desired measurement.
                Must have the same length as the number of MEASURE gates in source_circuit
                If self.n_shots is set, statistics are performed assuming self.n_shots successes
            save_mid_circuit_meas (bool): Save mid-circuit measurement results to
                self.mid_circuit_meas_freqs. All measurements will be saved to
                self.all_frequencies, with keys of length (n_meas + n_qubits).
                The leading n_meas values will hold the results of the MEASURE gates,
                ordered by their appearance in the source_circuit.
                The last n_qubits values will hold the measurements performed on
                each of qubits at the end of the circuit.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        n_meas = source_circuit.counts.get("MEASURE", 0)

        if desired_meas_result is not None:
            if not isinstance(desired_meas_result, str) or len(desired_meas_result) != n_meas:
                raise ValueError("desired_meas result is not a string with the same length as the number of measurements "
                                 "in the circuit.")
            save_mid_circuit_meas = True
        elif save_mid_circuit_meas and return_statevector:
            if self.n_shots != 1:
                raise ValueError("The combination of save_mid_circuit_meas and return_statevector without specifying desired_meas_result "
                                 "is only valid for self.n_shots=1. The result is a mixed state otherwise, "
                                 f"but you requested n_shots={self.n_shots}.")
        elif source_circuit.is_mixed_state and not self.n_shots:
            raise ValueError("Circuit contains MEASURE instruction, and is assumed to prepare a mixed state. "
                             "Please set the n_shots attribute to an appropriate value.")

        if source_circuit.width == 0:
            raise ValueError("Cannot simulate an empty circuit (e.g identity unitary) with unknown number of qubits.")

        # If the unitary is the identity (no gates) and no noise model, no need for simulation:
        # return all-zero state or sample from statevector
        if source_circuit.size == 0 and not self._noise_model:
            if initial_statevector is not None:
                statevector = initial_statevector
                frequencies = self._statevector_to_frequencies(initial_statevector)
            else:
                frequencies = {'0'*source_circuit.width: 1.0}
                statevector = np.zeros(2**source_circuit.width)
                statevector[0] = 1.0
            return (frequencies, statevector) if return_statevector else (frequencies, None)

        # For mid-circuit measurements post-process the result
        if save_mid_circuit_meas:
            # TODO: refactor to break a circular import. May involve by relocating get_xxx_oneterm functions
            from tangelo.toolboxes.post_processing.post_selection import split_frequency_dict

            (all_frequencies, statevector) = self.simulate_circuit(source_circuit,
                                                                   return_statevector=return_statevector,
                                                                   initial_statevector=initial_statevector,
                                                                   desired_meas_result=desired_meas_result,
                                                                   save_mid_circuit_meas=save_mid_circuit_meas)
            self.mid_circuit_meas_freqs, frequencies = split_frequency_dict(all_frequencies,
                                                                            list(range(n_meas)),
                                                                            desired_measurement=desired_meas_result)
            return (frequencies, statevector)

        return self.simulate_circuit(source_circuit,
                                     return_statevector=return_statevector,
                                     initial_statevector=initial_statevector)

    def get_expectation_value(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a quantum circuit preparing a
        state |\psi>. Return the expectation value <\psi | H | \psi>.

        In the case of a noiseless simulation, if the target backend exposes the
        statevector then it is used directly to compute expectation values, or
        draw samples if required. In the case of a noisy simulator, or if the
        statevector is not available on the target backend, individual shots
        must be run and the workflow is akin to what we would expect from an
        actual QPU.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (array): The initial statevector for the simulation
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """
        # Check if simulator supports statevector
        if initial_statevector is not None and not self.statevector_available:
            raise ValueError(f'Statevector not supported in {self.__class__}')

        # Check that qubit operator does not operate on qubits beyond circuit size.
        # Keep track if coefficients are real or not
        are_coefficients_real = True
        for term, coef in qubit_operator.terms.items():
            if state_prep_circuit.width < len(term):
                raise ValueError(f'Term {term} requires more qubits than the circuit contains ({state_prep_circuit.width})')
            if type(coef) in {complex, np.complex64, np.complex128}:
                are_coefficients_real = False

        # If the underlying operator is hermitian, expectation value is real and can be computed right away
        if are_coefficients_real:
            if self._noise_model or not self.statevector_available \
                    or (state_prep_circuit.is_mixed_state and self.n_shots is not None) or state_prep_circuit.size == 0:
                return self._get_expectation_value_from_frequencies(qubit_operator,
                                                                    state_prep_circuit,
                                                                    initial_statevector=initial_statevector,
                                                                    desired_meas_result=desired_meas_result)
            elif self.statevector_available:
                return self._get_expectation_value_from_statevector(qubit_operator,
                                                                    state_prep_circuit,
                                                                    initial_statevector=initial_statevector,
                                                                    desired_meas_result=desired_meas_result)

        # Else, separate the operator into 2 hermitian operators, use linearity and call this function twice
        else:
            qb_op_real, qb_op_imag = QubitOperator(), QubitOperator()
            for term, coef in qubit_operator.terms.items():
                qb_op_real.terms[term], qb_op_imag.terms[term] = coef.real, coef.imag
            qb_op_real.compress()
            qb_op_imag.compress()
            exp_real = self.get_expectation_value(qb_op_real, state_prep_circuit, initial_statevector=initial_statevector,
                                                  desired_meas_result=desired_meas_result)
            exp_imag = self.get_expectation_value(qb_op_imag, state_prep_circuit, initial_statevector=initial_statevector,
                                                  desired_meas_result=desired_meas_result)
            return exp_real if (exp_imag == 0.) else exp_real + 1.0j * exp_imag

    def get_variance(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a quantum circuit preparing a
        state |\psi>. Return the variance <\psi | H | \psi>.

        In the case of a noiseless simulation, if the target backend exposes the
        statevector then it is used directly to compute variance (zero), or
        draw samples if required. In the case of a noisy simulator, or if the
        statevector is not available on the target backend, individual shots
        must be run and the workflow is akin to what we would expect from an
        actual QPU.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The variance of this operator with regard to the
                state preparation.
        """
        # Check if simulator supports statevector
        if initial_statevector is not None and not self.statevector_available:
            raise ValueError(f'Statevector not supported in {self.__class__}')

        # Check that qubit operator does not operate on qubits beyond circuit size.
        # Keep track if coefficients are real or not
        are_coefficients_real = True
        for term, coef in qubit_operator.terms.items():
            if state_prep_circuit.width < len(term):
                raise ValueError(f'Term {term} requires more qubits than the circuit contains ({state_prep_circuit.width})')
            if type(coef) in {complex, np.complex64, np.complex128}:
                are_coefficients_real = False

        # If the underlying operator is hermitian, expectation value is real and can be computed right away
        if are_coefficients_real:
            return self._get_variance_from_frequencies(qubit_operator,
                                                       state_prep_circuit,
                                                       initial_statevector=initial_statevector,
                                                       desired_meas_result=desired_meas_result)

        # Else, separate the operator into 2 hermitian operators, use linearity and call this function twice
        else:
            qb_op_real, qb_op_imag = QubitOperator(), QubitOperator()
            for term, coef in qubit_operator.terms.items():
                qb_op_real.terms[term], qb_op_imag.terms[term] = coef.real, coef.imag
            qb_op_real.compress()
            qb_op_imag.compress()
            var_real = self.get_variance(qb_op_real, state_prep_circuit, initial_statevector=initial_statevector)
            var_imag = self.get_variance(qb_op_imag, state_prep_circuit, initial_statevector=initial_statevector)
            # https://en.wikipedia.org/wiki/Complex_random_variable#Variance_and_pseudo-variance
            return var_real if (var_imag == 0.) else var_real + var_imag  # always non-negative real number

    def get_standard_error(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a quantum circuit preparing a
        state |\psi>. Return the standard error of <\psi | H | \psi>, e.g. sqrt(Var H / n_shots).

        In the case of a noiseless simulation, if the target backend exposes the
        statevector then it is used directly to compute standard error (zero), or
        draw samples if required. In the case of a noisy simulator, or if the
        statevector is not available on the target backend, individual shots
        must be run and the workflow is akin to what we would expect from an
        actual QPU.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (list/array): A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The standard error of this operator with regard to the
                state preparation.
        """
        variance = self.get_variance(qubit_operator, state_prep_circuit, initial_statevector, desired_meas_result=desired_meas_result)
        return np.sqrt(variance/self.n_shots) if self.n_shots else 0.

    def _get_expectation_value_from_statevector(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi>, computed
        without drawing samples (statevector only). Users should not be calling
        this function directly, please call "get_expectation_value" instead.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation (only pure states).
            initial_statevector (array): The initial state of the system

        Returns:
            complex: The expectation value of this operator with regards to the
                state preparation.
        """

        n_qubits = state_prep_circuit.width

        expectation_value = 0.
        prepared_frequencies, prepared_state = self.simulate(state_prep_circuit, return_statevector=True,
                                                             initial_statevector=initial_statevector, desired_meas_result=desired_meas_result)

        if hasattr(self, "expectation_value_from_prepared_state"):
            return self.expectation_value_from_prepared_state(qubit_operator, n_qubits, prepared_state)

        # Otherwise, use generic statevector expectation value
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:  # Cannot have a qubit index beyond circuit size
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            if not self.n_shots:
                # Directly simulate and compute expectation value using statevector
                pauli_circuit = Circuit([Gate(pauli, index) for index, pauli in term], n_qubits=n_qubits)
                _, pauli_state = self.simulate(pauli_circuit, return_statevector=True, initial_statevector=prepared_state)

                delta = np.dot(pauli_state.real, prepared_state.real) + np.dot(pauli_state.imag, prepared_state.imag)
                expectation_value += coef * delta

            else:
                # Run simulation with statevector but compute expectation value with samples directly drawn from it
                basis_circuit = Circuit(measurement_basis_gates(term), n_qubits=state_prep_circuit.width)
                if basis_circuit.size > 0:
                    frequencies, _ = self.simulate(basis_circuit, initial_statevector=prepared_state)
                else:
                    frequencies = prepared_frequencies
                expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
                expectation_value += coef * expectation_term

        return expectation_value

    def _get_expectation_value_from_frequencies(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the expectation value <\psi | H | \psi> computed
        using the frequencies of observable states.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (array): The initial state of the system
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The expectation value of this operator with regard to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width
        if not self.statevector_available or state_prep_circuit.is_mixed_state or self._noise_model:
            initial_circuit = state_prep_circuit
            if initial_statevector is not None and not self.statevector_available:
                raise ValueError(f'Backend {self.__class__} does not support statevectors')
            else:
                updated_statevector = initial_statevector
        else:
            initial_circuit = Circuit(n_qubits=n_qubits)
            _, updated_statevector = self.simulate(state_prep_circuit,
                                                   return_statevector=True,
                                                   initial_statevector=initial_statevector,
                                                   desired_meas_result=desired_meas_result)

        expectation_value = 0.
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                expectation_value += coef
                continue

            basis_circuit = Circuit(measurement_basis_gates(term))
            full_circuit = initial_circuit + basis_circuit if (basis_circuit.size > 0) else initial_circuit
            frequencies, _ = self.simulate(full_circuit,
                                           initial_statevector=updated_statevector,
                                           desired_meas_result=desired_meas_result)
            expectation_term = self.get_expectation_value_from_frequencies_oneterm(term, frequencies)
            expectation_value += coef * expectation_term

        return expectation_value

    def _get_variance_from_frequencies(self, qubit_operator, state_prep_circuit, initial_statevector=None, desired_meas_result=None):
        r"""Take as input a qubit operator H and a state preparation returning a
        ket |\psi>. Return the variance of <\psi | H | \psi> computed
        using the frequencies of observable states.

        Args:
            qubit_operator (QubitOperator): the qubit operator.
            state_prep_circuit (Circuit): an abstract circuit used for state preparation.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The mid-circuit measurement results to select for.

        Returns:
            complex: The variance of this operator with regard to the
                state preparation.
        """
        n_qubits = state_prep_circuit.width
        if not self.statevector_available or state_prep_circuit.is_mixed_state or self._noise_model:
            initial_circuit = state_prep_circuit
            if initial_statevector is not None and not self.statevector_available:
                raise ValueError(f'Backend {self.__class__} does not support statevectors')
            else:
                updated_statevector = initial_statevector
        else:
            initial_circuit = Circuit(n_qubits=n_qubits)
            _, updated_statevector = self.simulate(state_prep_circuit,
                                                   return_statevector=True,
                                                   initial_statevector=initial_statevector,
                                                   desired_meas_result=desired_meas_result)

        variance = 0.
        for term, coef in qubit_operator.terms.items():

            if len(term) > n_qubits:
                raise ValueError(f"Size of operator {qubit_operator} beyond circuit width ({n_qubits} qubits)")
            elif not term:  # Empty term: no simulation needed
                pass

            basis_circuit = Circuit(measurement_basis_gates(term))
            full_circuit = initial_circuit + basis_circuit if (basis_circuit.size > 0) else initial_circuit
            frequencies, _ = self.simulate(full_circuit, initial_statevector=updated_statevector)
            variance_term = self.get_variance_from_frequencies_oneterm(term, frequencies)
            # Assumes no correlation between terms
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
            variance += coef * coef * variance_term

        return variance

    @staticmethod
    def get_expectation_value_from_frequencies_oneterm(term, frequencies):
        """Return the expectation value of a single-term qubit-operator, given
        the result of a state-preparation.

        Args:
            term (QubitOperator): a single-term qubit operator
            frequencies (dict): histogram of frequencies of measurements (assumed
                to be in lsq-first format).

        Returns:
            complex: The expectation value of this operator with regard to the
                state preparation.
        """

        return get_expectation_value_from_frequencies_oneterm(term, frequencies)

    @staticmethod
    def get_variance_from_frequencies_oneterm(term, frequencies):
        """Return the variance of a single-term qubit-operator, given
        the result of a state-preparation.

        Args:
            term (QubitOperator): a single-term qubit operator.
            frequencies (dict): histogram of frequencies of measurements (assumed
                to be in lsq-first format).

        Returns:
            complex: The variance of this operator with regard to the
                state preparation.
        """

        return get_variance_from_frequencies_oneterm(term, frequencies)

    def _statevector_to_frequencies(self, statevector):
        """For a given statevector representing the quantum state of a qubit
        register, returns a sparse histogram of the probabilities in the
        least-significant-qubit (lsq) -first order. e.g the string '100' means
        qubit 0 measured in basis state |1>, and qubit 1 & 2 both measured in
        state |0>.

        Args:
            statevector (list or ndarray(complex)): an iterable 1D data-structure
                containing the amplitudes.

        Returns:
            dict: A dictionary whose keys are bitstrings representing the
                multi-qubit states with the least significant qubit first (e.g
                '100' means qubit 0 in state |1>, and qubit 1 and 2 in state
                |0>), and the associated value is the corresponding frequency.
                Unless threshold=0., this dictionary will be sparse.
        """

        n_qubits = int(math.log2(len(statevector)))
        frequencies = dict()
        for i, amplitude in enumerate(statevector):
            frequency = abs(amplitude)**2
            if (frequency - self.freq_threshold) >= 0.:
                frequencies[self._int_to_binstr(i, n_qubits)] = frequency

        # If n_shots, has been specified, then draw that amount of samples from the distribution
        # and return empirical frequencies instead. Otherwise, return the exact frequencies
        if not self.n_shots:
            return frequencies
        else:
            xk, pk = [], []
            for k, v in frequencies.items():
                xk.append(int(k[::-1], 2))
                pk.append(frequencies[k])
            distr = stats.rv_discrete(name='distr', values=(np.array(xk), np.array(pk)))

            # Generate samples from distribution. Cut in chunks to ensure samples fit in memory, gradually accumulate
            chunk_size = 10**7
            n_chunks = self.n_shots // chunk_size
            freqs_shots = Counter()

            for i in range(n_chunks+1):
                this_chunk = self.n_shots % chunk_size if i == n_chunks else chunk_size
                samples = distr.rvs(size=this_chunk)
                freqs_shots += Counter(samples)
            freqs_shots = {self._int_to_binstr(k, n_qubits, False): v / self.n_shots for k, v in freqs_shots.items()}
            return freqs_shots

    def _int_to_binstr(self, i, n_qubits, use_ordering=True):
        """Convert an integer into a bit string of size n_qubits.

        Args:
            i (int): integer to convert to bit string.
            n_qubits (int): The number of qubits and length of returned bit string.
            use_ordering (bool): Flip the order of the returned bit string
                depending on self.statevector_order being "msq_first" or "lsq_first"

        Returns:
            string: The bit string of the integer in lsq-first order.
        """
        bs = bin(i).split('b')[-1]
        state_binstr = "0" * (n_qubits - len(bs)) + bs

        return state_binstr if use_ordering and (self.statevector_order == "lsq_first") else state_binstr[::-1]

    def collapse_statevector_to_desired_measurement(self, statevector, qubit, result):
        """Take 0 or 1 part of a statevector for a given qubit and return a normalized statevector and probability.

        Args:
            statevector (array): The statevector for which the collapse to the desired qubit value is performed.
            qubit (int): The index of the qubit to collapse to the classical result.
            result (string): "0" or "1".

        Returns:
            array: the collapsed and renormalized statevector
            float: the probability this occured
        """

        return collapse_statevector_to_desired_measurement(statevector, qubit, result, self.backend_info()['statevector_order'])

    @staticmethod
    @abc.abstractmethod
    def backend_info() -> dict:
        """A dictionary that includes {'noisy_simulation': True or False,
                                       'statevector_available': True or False,
                                       'statevector_order': 'lsq_first' or 'msq_first'"""
        pass
