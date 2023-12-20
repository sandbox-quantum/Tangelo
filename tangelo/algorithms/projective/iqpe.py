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

"""Implements the iterative Quantum Phase Estimation (iQPE) algorithm to solve
electronic structure calculations.

Ref:
    M. Dobsicek, G. Johansson, V. Shumeiko, and G. Wendin, Arbitrary Accuracy Iterative Quantum Phase
    Estimation Algorithm using a Single Ancillary Qubit: A two-qubit benchmark, Phys. Rev. A 76, 030306(R)
    (2007).
"""

from typing import Optional, Union, List
from collections import Counter
from enum import Enum

import numpy as np

from tangelo import SecondQuantizedMolecule
from tangelo.linq import get_backend, Circuit, Gate, ClassicalControl, generate_applied_gates
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_mapped_vector, vector_to_circuit, get_reference_circuit
import tangelo.toolboxes.unitary_generator as ugen
import tangelo.toolboxes.ansatz_generator as agen
from tangelo.toolboxes.post_processing.histogram import Histogram


class BuiltInUnitary(Enum):
    """Enumeration of the ansatz circuits supported by iQPE."""
    TrotterSuzuki = ugen.TrotterSuzukiUnitary
    CircuitUnitary = ugen.CircuitUnitary


class IterativeQPESolver:
    r"""Solve the electronic structure problem for a molecular system by using
    the iterative Quantum Phase Estimation (iQPE) algorithm.

    This algorithm evaluates the energy of a molecular system by performing
    a series of controlled time-evolutions

    Users must first set the desired options of the iterative QPESolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, unitary...). They are then able to call the
    the simulate method. In particular, simulate
    runs the iQPE algorithm, returning the optimal energy found by the most probable
    measurement as a binary fraction.

    Attributes:
        molecule (SecondQuantizedMolecule) : The molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        unitary (Unitary) : one of the supported unitary evolutions.
        backend_options (dict): parameters to build the underlying compute backend (simulator, etc).
        simulate_options (dict): Options for fine-control of the simulator backend, including desired measurement results, etc.
        penalty_terms (dict): parameters for penalty terms to append to target
            qubit Hamiltonian (see penalty_terms for more details).
        unitary_options (dict): parameters for the given ansatz (see given ansatz
            file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False has alternating
                spin up/down ordering.
        qubit_hamiltonian (QubitOperator): The Hamiltonian expressed as a sum of products of Pauli matrices.
        verbose (bool): Flag for iQPE verbosity.
        projective_circuit (Circuit): A terminal circuit that projects into the correct space, always added to
            the end of the unitary circuit. Could be measurement gates for example
        ref_state (array or Circuit): The reference configuration to use. Replaces HF state
        size_qpe_register (int): The number of iterations of single qubit iQPE to use for the calculation.
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": 1, "noise_model": None}
        copt_dict = opt_dict.copy()

        self.molecule: Optional[SecondQuantizedMolecule] = copt_dict.pop("molecule", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.unitary: ugen.Unitary = copt_dict.pop("unitary", BuiltInUnitary.TrotterSuzuki)
        self.backend_options: dict = copt_dict.pop("backend_options", default_backend_options)
        self.penalty_terms: Optional[dict] = copt_dict.pop("penalty_terms", None)
        self.simulate_options: dict = copt_dict.pop("simulate_options", dict())
        self.unitary_options: dict = copt_dict.pop("unitary_options", dict())
        self.up_then_down: bool = copt_dict.pop("up_then_down", False)
        self.qubit_hamiltonian: QubitOperator = copt_dict.pop("qubit_hamiltonian", None)
        self.verbose: bool = copt_dict.pop("verbose", False)
        self.projective_circuit: Circuit = copt_dict.pop("projective_circuit", None)
        self.ref_state: Optional[Union[list, Circuit]] = copt_dict.pop("ref_state", None)
        self.n_qpe_qubits: int = copt_dict.pop("size_qpe_register", 1)

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        # If nothing is provided raise an Error:
        if not (bool(self.molecule) or bool(self.qubit_hamiltonian) or isinstance(self.unitary, (ugen.Unitary, Circuit))):
            raise ValueError(f"A Molecule or a QubitOperator or a Unitary object/Circuit must be provided in {self.__class__.__name__}")

        # Raise error/warnings if input is not as expected. Only a single input
        if (bool(self.molecule) and bool(self.qubit_hamiltonian)):
            raise ValueError(f"Incompatible Options in {self.__class__.__name__}:"
                             "Only one of the following can be provided by user: molecule OR qubit Hamiltonian.")
        if isinstance(self.unitary, (Circuit, ugen.Unitary)) and bool(self.qubit_hamiltonian):
            raise ValueError(f"Incompatible Options in {self.__class__.__name__}:"
                             "Only one of the following can be provided by user: unitary OR qubit Hamiltonian.")
        if isinstance(self.unitary, (Circuit, ugen.Unitary)) and bool(self.molecule):
            raise Warning("The molecule is only being used to generate the reference state. The unitary is being used for the iQPE.")

        # Initialize the reference state circuit.
        if self.ref_state is not None:
            if isinstance(self.ref_state, Circuit):
                self.reference_circuit = self.ref_state
            else:
                self.reference_circuit = vector_to_circuit(get_mapped_vector(self.ref_state, self.qubit_mapping, self.up_then_down))
        else:
            if bool(self.molecule):
                self.reference_circuit = get_reference_circuit(self.molecule.n_active_sos,
                                                               self.molecule.n_active_electrons,
                                                               self.qubit_mapping,
                                                               self.up_then_down,
                                                               self.molecule.spin)
            else:
                self.reference_circuit = Circuit()

        default_backend_options.update(self.backend_options)
        self.backend_options = default_backend_options
        self.builtin_unitary = set(BuiltInUnitary)

    def build(self):
        """Build the underlying objects required to run the iQPE algorithm
        afterwards.
        """

        if isinstance(self.unitary, Circuit):
            self.unitary = ugen.CircuitUnitary(self.unitary, **self.unitary_options)

        # Building QPE with a molecule as input.
        if self.molecule:

            # Compute qubit hamiltonian for the input molecular system
            self.qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                              mapping=self.qubit_mapping,
                                                              n_spinorbitals=self.molecule.n_active_sos,
                                                              n_electrons=self.molecule.n_active_electrons,
                                                              up_then_down=self.up_then_down,
                                                              spin=self.molecule.active_spin)

            if self.penalty_terms:
                pen_ferm = agen.penalty_terms.combined_penalty(self.molecule.n_active_mos, self.penalty_terms)
                pen_qubit = fermion_to_qubit_mapping(fermion_operator=pen_ferm,
                                                     mapping=self.qubit_mapping,
                                                     n_spinorbitals=self.molecule.n_active_sos,
                                                     n_electrons=self.molecule.n_active_electrons,
                                                     up_then_down=self.up_then_down,
                                                     spin=self.molecule.active_spin)
                self.qubit_hamiltonian += pen_qubit

        if isinstance(self.unitary, BuiltInUnitary):
            self.unitary = self.unitary.value(self.qubit_hamiltonian, **self.unitary_options)
        elif not isinstance(self.unitary, ugen.Unitary):
            raise TypeError("Invalid ansatz dataype. Expecting a custom Unitary (Unitary class).")

        # Quantum circuit simulation backend options
        self.backend = get_backend(**self.backend_options)

        # Determine where to place QPE ancilla qubit index
        self.n_state, self.n_ancilla = self.unitary.qubit_indices()
        self.qft_qubit = max(list(self.n_state)+list(self.n_ancilla)) + 1

        self.cfunc = IterativeQPEControl(self.n_qpe_qubits, self.qft_qubit, self.unitary)
        self.circuit = Circuit(self.reference_circuit._gates+[Gate("CMEASURE", self.qft_qubit)],
                               cmeasure_control=self.cfunc, n_qubits=self.qft_qubit+1)

    def simulate(self):
        """Run the iQPE circuit. Return the energy of the most probable bitstring

        Attributes:
            bitstring (str): The most probable bitstring.
            histogram (Histogram): The full Histogram of measurements on the iQPE ancilla qubit representing energies.
            qpe_freqs (dict): The dictionary of measurements on the iQPE ancilla qubit.
            freqs (dict): The full dictionary of measurements on all qubits.

        Returns:
            float: The energy of the most probable bitstring measured during iQPE.
        """

        if not (self.unitary and self.backend):
            raise RuntimeError(f"No unitary or hardware backend built. Have you called {self.__class__.__name__}.build ?")

        self.freqs, _ = self.backend.simulate(self.circuit)
        self.histogram = Histogram(self.freqs)
        self.histogram.remove_qubit_indices(*(self.n_state+self.n_ancilla))
        qpe_counts = Counter(self.cfunc.measurements[:self.backend.n_shots])
        self.qpe_freqs = {key[::-1]: value/self.backend.n_shots for key, value in qpe_counts.items()}
        self.bitstring = max(self.qpe_freqs.items(), key=lambda x: x[1])[0]

        return self.energy_estimation(self.bitstring)

    def get_resources(self):
        """Estimate the resources required by iQPE, with the current unitary. This
        assumes "build" has been run, as it requires the circuit and the
        qubit Hamiltonian. Return information that pertains to the user, for the
        purpose of running an experiment on a classical simulator or a quantum
        device.
        """

        resources = dict()

        # If the attribute of the applied_gates has been populated, use the exact resources else approximate the iQPE circuit
        circuit = Circuit(self.circuit.applied_gates) if self.circuit.applied_gates else Circuit(generate_applied_gates(self.circuit))
        resources["applied_circuit_width"] = circuit.width
        resources["applied_circuit_depth"] = circuit.depth()
        resources["applied_circuit_2qubit_gates"] = circuit.counts_n_qubit.get(2, 0)
        return resources

    def energy_estimation(self, bitstring):
        """Estimate energy using the calculated frequency dictionary.

        Args:
            bitstring (str): The bitstring to evaluate the energy of in base 2.

        Returns:
             float: Energy of the given bitstring
        """

        return sum([0.5**(i+1) for i, b in enumerate(bitstring) if b == "1"])


class IterativeQPEControl(ClassicalControl):
    def __init__(self, n_bits: int, qft_qubit: int, u: ugen.Unitary):
        """Iterative QPE with n_bits"""
        self.n_bits: int = n_bits
        self.bitplace: int = n_bits
        self.phase: float = 0
        self.measurements: List[str] = [""]
        self.energies: List[float] = [0.]
        self.n_runs: int = 0
        self.qft_qubit: int = qft_qubit
        self.unitary: ugen.Unitary = u
        self.started: bool = False

    def return_gates(self, measurement) -> List[Gate]:
        """Return a list of gates based on the measurement outcome for the next step in iQPE.

        Each measurement updates the current phase correction and returns a list of gates that
        implements the next controlled time-evolution along with the phase correction to determine
        the next bit value.

        Args:
            measurement (str): "1" or "0"

        Returns:
            List[Gate]: The next gates to apply to the circuit
        """
        # Ignore the first measurement as it is always 0 and meaningless.
        if self.started:
            self.measurements[self.n_runs] += measurement
            self.energies[self.n_runs] += int(measurement)/2**self.bitplace
        else:
            self.started = True

        if self.bitplace > 0:
            # Update phase and determine reset gates
            if measurement == "1":
                self.phase += 1/2**(self.bitplace)
                reset_to_zero = [Gate("X", self.qft_qubit)]
            else:
                reset_to_zero = []

            # Decrease bitplace and apply next phase estimation
            self.bitplace -= 1
            phase_correction = [Gate("PHASE", self.qft_qubit, parameter=-np.pi*self.phase*2**(self.bitplace))]
            gates = reset_to_zero + [Gate("H", self.qft_qubit)] + phase_correction
            gates += self.unitary.build_circuit(2**(self.bitplace), self.qft_qubit)._gates + [Gate("H", self.qft_qubit)]
            return gates + [Gate("CMEASURE", self.qft_qubit)]
        else:
            return []

    def finalize(self):
        """Called from simulator after all gates have been called.

        Reinitialize all variables and store measurements and energies from the current iQPE run.
        """
        self.bitplace = self.n_bits
        self.phase = 0
        self.n_runs += 1
        self.measurements += [""]
        self.energies += [0.]
        self.started = False
