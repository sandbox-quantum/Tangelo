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

"""Implements the variational quantum eigensolver (VQE) algorithm to solve
electronic structure calculations.
"""

from typing import Optional, Union

from enum import Enum

from tangelo.helpers.utils import HiddenPrints
from tangelo import SecondQuantizedMolecule
from tangelo.linq import get_backend, Circuit
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_mapped_vector, vector_to_circuit, get_reference_circuit
import tangelo.toolboxes.unitary_generator as unitary
import tangelo.toolboxes.ansatz_generator as agen
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.toolboxes.post_processing.histogram import Histogram


class BuiltInUnitary(Enum):
    """Enumeration of the ansatz circuits supported by QPE."""
    TrotterSuzuki = unitary.TrotterSuzukiUnitary
    CircuitUnitary = unitary.CircuitUnitary


class QPESolver:
    r"""Solve the electronic structure problem for a molecular system by using
    the Quantum Phase Estimation (QPE) algorithm.

    This algorithm evaluates the energy of a molecular system by performing
    a series of controlled time-evolutions

    Users must first set the desired options of the QPESolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the simulate, or get_rdm methods. In particular, simulate
    runs the VQE algorithm, returning the optimal energy found by the classical
    optimizer.

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
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
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        verbose (bool): Flag for QPE verbosity.
        projective_circuit (Circuit): A terminal circuit that projects into the correct space, always added to
            the end of the unitary circuit. Could be measurement gates for example
        ref_state (array or Circuit): The reference configuration to use. Replaces HF state
        size_qpe_register (int): The number of qubits to use for the qpe register
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        copt_dict = opt_dict.copy()

        self.molecule: Optional[SecondQuantizedMolecule] = copt_dict.pop("molecule", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.unitary: unitary.Unitary = copt_dict.pop("unitary", BuiltInUnitary.TrotterSuzuki)
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

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

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
        """Build the underlying objects required to run the VQE algorithm
        afterwards.
        """

        if isinstance(self.unitary, Circuit):
            self.unitary = unitary.CircuitUnitary(self.unitary)

        # Building VQE with a molecule as input.
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

            # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
            if self.unitary in {BuiltInUnitary.TrotterSuzuki}:
                self.unitary = self.unitary.value(self.qubit_hamiltonian, **self.unitary_options)
        if isinstance(self.unitary, BuiltInUnitary):
            self.unitary = self.unitary.value(self.qubit_hamiltonian, **self.unitary_options)
        elif not isinstance(self.unitary, unitary.Unitary):
            raise TypeError(f"Invalid ansatz dataype. Expecting a custom Unitary (Unitary class).")

        # Quantum circuit simulation backend options
        self.backend = get_backend(**self.backend_options)

        self.circuit = Circuit()

        self.n_state, self.n_ancilla = self.unitary.qubit_indices()
        qft_start = max(list(self.n_state)+list(self.n_ancilla)) + 1
        self.qpe_qubit_list = list(reversed(range(qft_start, qft_start+self.n_qpe_qubits)))
        qft = get_qft_circuit(self.qpe_qubit_list)

        self.circuit = qft
        for i, qubit in enumerate(self.qpe_qubit_list):
            self.circuit += self.unitary.build_circuit(2**i, control=qubit)
        iqft = get_qft_circuit(self.qpe_qubit_list, inverse=True)
        self.circuit += iqft

    def simulate(self):
        """Run the QPE circuit. Return the energy of the most probable bitstring.
        """
        if not (self.unitary and self.backend):
            raise RuntimeError("No ansatz circuit or hardware backend built. Have you called VQESolver.build ?")

        self.freqs, _ = self.backend.simulate(self.reference_circuit+self.circuit)
        self.histogram = Histogram(self.freqs)
        self.histogram.remove_qubit_indices(*(self.n_state+self.n_ancilla))
        self.qpe_freqs = self.histogram.frequencies
        self.key_max = max(zip(self.qpe_freqs.values(), self.qpe_freqs.keys()))

        return self.energy_estimation(self.key_max[1])

    def get_resources(self):
        """Estimate the resources required by QPE, with the current unitary. This
        assumes "build" has been run, as it requires the circuit and the
        qubit Hamiltonian. Return information that pertains to the user, for the
        purpose of running an experiment on a classical simulator or a quantum
        device.
        """

        resources = dict()
        resources["qubit_hamiltonian_terms"] = len(self.qubit_hamiltonian.terms)
        circuit = self.circuit if self.ref_state is None else self.reference_circuit + self.circuit
        resources["circuit_width"] = circuit.width
        resources["circuit_depth"] = circuit.depth()
        resources["circuit_2qubit_gates"] = circuit.counts_n_qubit.get(2, 0)
        return resources

    def energy_estimation(self, bitstring):
        """Estimate energy using the calculated frequency dictionary.

        Args:
            bitstring (str): The bitstring to evaluate the energy of in base 10.

        Returns:
             float: Energy of the given bitstring
        """

        energy = 0.
        for i, b in enumerate(bitstring):
            if b == "1":
                energy += 1/2**(i+1)

        return energy

        """Function used as a default optimizer for VQE when user does not
        provide one. Can be used as an example for users who wish to provide
        their custom optimizer.

        Should set the attributes "optimal_var_params" and "optimal_energy" to
        ensure the outcome of VQE is captured at the end of classical
        optimization, and can be accessed in a standard way.

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes var_params as input and returns
                a float.
            var_params (list): The variational parameters (float64).

        Returns:
            float: The optimal energy found by the optimizer.
            list of floats: Optimal parameters
        """

        from scipy.optimize import minimize

        with HiddenPrints():
            result = minimize(func, var_params, method="SLSQP",
                              options={"disp": True, "maxiter": 2000, "eps": 1e-5, "ftol": 1e-5})

        if self.verbose:
            print(f"VQESolver optimization results:")
            print(f"\tOptimal VQE energy: {result.fun}")
            print(f"\tOptimal VQE variational parameters: {result.x}")
            print(f"\tNumber of Iterations : {result.nit}")
            print(f"\tNumber of Function Evaluations : {result.nfev}")
            print(f"\tNumber of Gradient Evaluations : {result.njev}")

        return result.fun, result.x
