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

import warnings
import itertools
from typing import Optional, Union, List

from enum import Enum
import numpy as np

from tangelo.helpers.utils import HiddenPrints
from tangelo import SecondQuantizedMolecule
from tangelo.linq import get_backend, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
from tangelo.toolboxes.operators import count_qubits, FermionOperator, QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_mapped_vector, vector_to_circuit
from tangelo.toolboxes.post_processing.bootstrapping import get_resampled_frequencies
from tangelo.toolboxes.optimizers import rotosolve
import tangelo.toolboxes.ansatz_generator as agen


class BuiltInAnsatze(Enum):
    """Enumeration of the ansatz circuits supported by VQE."""
    UCCSD = agen.UCCSD
    UCC1 = agen.RUCC(1)
    UCC3 = agen.RUCC(3)
    HEA = agen.HEA
    UpCCGSD = agen.UpCCGSD
    QMF = agen.QMF
    QCC = agen.QCC
    VSQS = agen.VSQS
    UCCGD = agen.UCCGD
    ILC = agen.ILC
    pUCCD = agen.pUCCD


class VQESolver:
    r"""Solve the electronic structure problem for a molecular system by using
    the variational quantum eigensolver (VQE) algorithm.

    This algorithm evaluates the energy of a molecular system by performing
    classical optimization over a parametrized ansatz circuit.

    Users must first set the desired options of the VQESolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the energy_estimation, simulate, or get_rdm methods. In particular, simulate
    runs the VQE algorithm, returning the optimal energy found by the classical
    optimizer.

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        ansatz (Ansatze) : one of the supported ansatze.
        optimizer (function handle): a function defining the classical optimizer
            and its behavior.
        initial_var_params (str or array-like) : initial value for the classical
            optimizer.
        backend_options (dict): parameters to build the underlying compute backend (simulator, etc).
        simulate_options (dict): Options for fine-control of the simulator backend, including desired measurement results, etc.
        penalty_terms (dict): parameters for penalty terms to append to target
            qubit Hamiltonian (see penalty_terms for more details).
        deflation_circuits (list[Circuit]): Deflation circuits to add an
            orthogonalization penalty with.
        deflation_coeff (float): The coefficient of the deflation.
        ansatz_options (dict): parameters for the given ansatz (see given ansatz
            file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False has alternating
                spin up/down ordering.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        verbose (bool): Flag for VQE verbosity.
        projective_circuit (Circuit): A terminal circuit that projects into the correct space, always added to
            the end of the ansatz circuit.
        ref_state (array or Circuit): The reference configuration to use. Replaces HF state
            QMF, QCC, ILC require ref_state to be an array. UCC1, UCC3, VSQS can not use a
            different ref_state than HF by construction.
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        copt_dict = opt_dict.copy()

        self.molecule: Optional[SecondQuantizedMolecule] = copt_dict.pop("molecule", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.ansatz: agen.Ansatz = copt_dict.pop("ansatz", BuiltInAnsatze.UCCSD)
        self.optimizer = copt_dict.pop("optimizer", self._default_optimizer)
        self.initial_var_params: Optional[Union[str, list]] = copt_dict.pop("initial_var_params", None)
        self.backend_options: dict = copt_dict.pop("backend_options", default_backend_options)
        self.penalty_terms: Optional[dict] = copt_dict.pop("penalty_terms", None)
        self.simulate_options: dict = copt_dict.pop("simulate_options", dict())
        self.deflation_circuits: Optional[List[Circuit]] = copt_dict.pop("deflation_circuits", list())
        self.deflation_coeff: float = copt_dict.pop("deflation_coeff", 1)
        self.ansatz_options: dict = copt_dict.pop("ansatz_options", dict())
        self.up_then_down: bool = copt_dict.pop("up_then_down", False)
        self.qubit_hamiltonian: QubitOperator = copt_dict.pop("qubit_hamiltonian", None)
        self.verbose: bool = copt_dict.pop("verbose", False)
        self.projective_circuit: Circuit = copt_dict.pop("projective_circuit", None)
        self.ref_state: Optional[Union[list, Circuit]] = copt_dict.pop("ref_state", None)

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

        # The QCC & ILC ansatze require up_then_down=True when mapping="jw"
        if isinstance(self.ansatz, BuiltInAnsatze):
            if self.ansatz in (BuiltInAnsatze.QCC, BuiltInAnsatze.ILC) and self.qubit_mapping.lower() == "jw" and not self.up_then_down:
                warnings.warn("Spin-orbital ordering shifted to all spin-up first then down to ensure efficient generator screening "
                              "for the Jordan-Wigner mapping with QCC-based ansatze.", RuntimeWarning)
                self.up_then_down = True
            if self.ansatz == BuiltInAnsatze.pUCCD and self.qubit_mapping.lower() != "hcb":
                warnings.warn("Forcing the hard-core boson mapping for the pUCCD ansatz.", RuntimeWarning)
                self.qubit_mapping = "HCB"
            # QCC and QMF and ILC require a reference state that can be represented by a single layer of RZ-RX gates on each qubit.
            # This decomposition can not be determined from a general Circuit reference state.
            if isinstance(self.ref_state, Circuit):
                if self.ansatz in [BuiltInAnsatze.QCC, BuiltInAnsatze.ILC, BuiltInAnsatze.QMF]:
                    raise ValueError("Circuit reference state is not supported for QCC or QMF")
            elif self.ref_state is not None:
                self.ansatz_options["reference_state"] = "zero"
                if self.ansatz in [BuiltInAnsatze.QCC, BuiltInAnsatze.ILC]:
                    self.ansatz_options["qmf_var_params"] = agen._qubit_mf.init_qmf_from_vector(self.ref_state, self.qubit_mapping, self.up_then_down)
                    self.ref_state = None
                elif self.ansatz == BuiltInAnsatze.QMF:
                    self.ansatz_options["init_qmf"] = {"init_params": "vector", "vector": self.ref_state}
                    self.ref_state = None
                # UCC1, UCC3, QMF and VSQS require the initial state to be Hartree-Fock.
                # UCC1 and UCC3 use a special structure
                # VSQS is only defined for a Hartree-Fock reference at this time
                elif self.ansatz in [BuiltInAnsatze.UCC1, BuiltInAnsatze.UCC3, BuiltInAnsatze.VSQS]:
                    raise ValueError("UCC1, UCC3, and VSQS do not support reference states other than Hartree-Fock at this time in Tangelo")

        if self.ref_state is not None:
            if isinstance(self.ref_state, Circuit):
                self.reference_circuit = self.ref_state
            else:
                self.reference_circuit = vector_to_circuit(get_mapped_vector(self.ref_state, self.qubit_mapping, self.up_then_down))
        else:
            self.reference_circuit = Circuit()

        default_backend_options.update(self.backend_options)
        self.backend_options = default_backend_options
        self.optimal_energy = None
        self.optimal_var_params = None
        self.builtin_ansatze = set(BuiltInAnsatze)

    def build(self):
        """Build the underlying objects required to run the VQE algorithm
        afterwards.
        """

        if isinstance(self.ansatz, Circuit):
            self.ansatz = agen.VariationalCircuitAnsatz(self.ansatz)

        # Check compatibility of optimizer with Ansatz class
        elif self.optimizer == rotosolve:
            if self.ansatz not in [BuiltInAnsatze.UCC1, BuiltInAnsatze.UCC3, BuiltInAnsatze.HEA]:
                raise ValueError(f"{self.ansatz} not compatible with rotosolve optimizer.")

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
                if self.ansatz == BuiltInAnsatze.QCC:
                    self.ansatz_options["qubit_ham"] = self.qubit_hamiltonian.to_qubitoperator()

            # Verification of system compatibility with UCC1 or UCC3 circuits.
            if self.ansatz in [BuiltInAnsatze.UCC1, BuiltInAnsatze.UCC3]:
                # Mapping should be JW because those ansatz are chemically inspired.
                if self.qubit_mapping.upper() != "JW":
                    raise ValueError("Qubit mapping must be JW for {} ansatz.".format(self.ansatz))
                # They are encoded with this convention.
                if not self.up_then_down:
                    raise ValueError("Parameter up_then_down must be set to True for {} ansatz.".format(self.ansatz))
                # Only HOMO-LUMO systems are relevant.
                if count_qubits(self.qubit_hamiltonian) != 4:
                    raise ValueError("The system must be reduced to a HOMO-LUMO problem for {} ansatz.".format(self.ansatz))

            # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
            if isinstance(self.ansatz, BuiltInAnsatze):
                if self.ansatz in {BuiltInAnsatze.UCC1, BuiltInAnsatze.UCC3}:
                    self.ansatz = self.ansatz.value
                elif self.ansatz == BuiltInAnsatze.pUCCD:
                    self.ansatz = self.ansatz.value(self.molecule, **self.ansatz_options)
                elif self.ansatz in self.builtin_ansatze:
                    self.ansatz = self.ansatz.value(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                else:
                    raise ValueError(f"Unsupported ansatz. Built-in ansatze:\n\t{self.builtin_ansatze}")
            elif not isinstance(self.ansatz, agen.Ansatz):
                raise TypeError(f"Invalid ansatz dataype. Expecting instance of Ansatz class, or one of built-in options:\n\t{self.builtin_ansatze}")

        # Building with a qubit Hamiltonian.
        elif self.ansatz in {BuiltInAnsatze.HEA, BuiltInAnsatze.VSQS}:
            self.ansatz = self.ansatz.value(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
        elif not isinstance(self.ansatz, agen.Ansatz):
            raise TypeError(f"Invalid ansatz dataype. Expecting a custom Ansatz (Ansatz class).")

        # Set ansatz initial parameters (default or use input), build corresponding ansatz circuit
        self.initial_var_params = self.ansatz.set_var_params(self.initial_var_params)
        self.ansatz.build_circuit()

        if len(self.ansatz.circuit._variational_gates) == 0:
            warnings.warn("No variational gate found in the circuit.", RuntimeWarning)

        # Quantum circuit simulation backend options
        self.backend = get_backend(**self.backend_options)

    def simulate(self):
        """Run the VQE algorithm, using the ansatz, classical optimizer, initial
        parameters and hardware backend built in the build method.
        """
        if not (self.ansatz and self.backend):
            raise RuntimeError("No ansatz circuit or hardware backend built. Have you called VQESolver.build ?")

        if len(self.ansatz.circuit._variational_gates) == 0:
            raise RuntimeError("No variational gate found in the circuit.")

        optimal_energy, optimal_var_params = self.optimizer(self.energy_estimation, self.initial_var_params)

        self.optimal_var_params = optimal_var_params
        self.optimal_energy = optimal_energy
        self.ansatz.build_circuit(self.optimal_var_params)
        self.optimal_circuit = self.reference_circuit+self.ansatz.circuit if self.ref_state is not None else self.ansatz.circuit
        if self.projective_circuit:
            self.optimal_circuit += self.projective_circuit
        return self.optimal_energy

    def get_resources(self):
        """Estimate the resources required by VQE, with the current ansatz. This
        assumes "build" has been run, as it requires the ansatz circuit and the
        qubit Hamiltonian. Return information that pertains to the user, for the
        purpose of running an experiment on a classical simulator or a quantum
        device.
        """

        resources = dict()
        resources["qubit_hamiltonian_terms"] = len(self.qubit_hamiltonian.terms) + len(self.deflation_circuits)
        circuit = self.ansatz.circuit if self.ref_state is None else self.reference_circuit + self.ansatz.circuit
        if self.deflation_circuits:
            circuit += self.deflation_circuits[0]
        resources["circuit_width"] = circuit.width
        resources["circuit_depth"] = circuit.depth()
        resources["circuit_2qubit_gates"] = circuit.counts_n_qubit.get(2, 0)
        resources["circuit_var_gates"] = len(self.ansatz.circuit._variational_gates)
        resources["vqe_variational_parameters"] = len(self.initial_var_params)
        return resources

    def energy_estimation(self, var_params):
        """Estimate energy using the given ansatz, qubit hamiltonian and compute
        backend. Keeps track of optimal energy and variational parameters along
        the way.

        Args:
             var_params (numpy.array or str): variational parameters to use for
                VQE energy evaluation.

        Returns:
             float: energy computed by VQE using the ansatz and input
                variational parameters.
        """

        # Update variational parameters, compute energy using the hardware backend
        self.ansatz.update_var_params(var_params)
        circuit = self.ansatz.circuit if self.ref_state is None else self.reference_circuit + self.ansatz.circuit
        if self.projective_circuit:
            circuit += self.projective_circuit
        energy = self.backend.get_expectation_value(self.qubit_hamiltonian, circuit, **self.simulate_options)

        # Additional computation for deflation (optional)
        for circ in self.deflation_circuits:
            f_dict, _ = self.backend.simulate(circ + circuit.inverse())
            energy += self.deflation_coeff * f_dict.get("0"*self.ansatz.circuit.width, 0)

        if self.verbose:
            print(f"\tEnergy = {energy:.7f} ")

        return energy

    def operator_expectation(self, operator, var_params=None, n_active_mos=None, n_active_electrons=None, n_active_sos=None, spin=None, ref_state=Circuit()):
        """Obtains the operator expectation value of a given operator.

           Args:
                operator (str or QubitOperator): The operator to find the
                    expectation value of str availability:
                    - N : Particle number
                    - Sz: Spin in z-direction
                    - S^2: Spin quantum number s(s+1)
                var_params (str or numpy.array): variational parameters to use
                    for VQE expectation value evaluation.
                n_active_mos (int): The number of active_mos (int). Only
                    required when using a str input and VQESolver is initiated
                    with a QubitHamiltonian.
                n_active_electrons (int): The number of active electrons. Only
                    required when operator is of type FermionOperator and
                    mapping used is scbk and vqe_solver was initiated using a
                    QubitHamiltonian.
                n_active_sos (int): Number of active spin orbitals. Only
                    required when operator is of type FermionOperator and
                    mapping used is scbk and vqe_solver was initiated using a
                    QubitHamiltonian.
                spin (int): Spin (n_alpha - n_beta)
                ref_state (Circuit): A reference state preparation circuit

           Returns:
                float: operator expectation value computed by VQE using the
                    ansatz and input variational parameters.
        """
        if var_params is None:
            var_params = self.ansatz.var_params

        # Save our current target hamiltonian
        tmp_hamiltonian = self.qubit_hamiltonian

        if isinstance(operator, str):
            if n_active_mos is None:
                if self.molecule:
                    n_active_mos = self.molecule.n_active_mos
                else:
                    raise KeyError("Must supply n_active_mos when a QubitHamiltonian has initialized VQESolver"
                                   " and requesting the expectation of 'N', 'Sz', or 'S^2'")
            if operator == "N":
                exp_op = agen.fermionic_operators.number_operator(n_active_mos, up_then_down=False)
            elif operator == "Sz":
                exp_op = agen.fermionic_operators.spinz_operator(n_active_mos, up_then_down=False)
            elif operator == "S^2":
                exp_op = agen.fermionic_operators.spin2_operator(n_active_mos, up_then_down=False)
            else:
                raise ValueError('Only expectation values of N, Sz and S^2')
        elif isinstance(operator, FermionOperator):
            exp_op = operator
        elif isinstance(operator, QubitOperator):
            self.qubit_hamiltonian = operator
        else:
            raise TypeError("operator must be a of string, FermionOperator or QubitOperator type.")

        if isinstance(operator, (str, FermionOperator)):
            if (n_active_electrons is None or n_active_sos is None or spin is None) and self.qubit_mapping == "scbk":
                if self.molecule:
                    n_active_electrons = self.molecule.n_active_electrons
                    n_active_sos = self.molecule.n_active_sos
                    spin = self.molecule.active_spin
                else:
                    raise KeyError("Must supply n_active_electrons, n_active_sos, and spin with a FermionOperator and scbk mapping.")

            self.qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=exp_op,
                                                              mapping=self.qubit_mapping,
                                                              n_spinorbitals=n_active_sos,
                                                              n_electrons=n_active_electrons,
                                                              up_then_down=self.up_then_down,
                                                              spin=spin)

        self.ansatz.update_var_params(var_params)
        circuit = ref_state + self.ansatz.circuit
        if self.projective_circuit:
            circuit += self.projective_circuit
        expectation = self.backend.get_expectation_value(self.qubit_hamiltonian, circuit, **self.simulate_options)

        # Restore the current target hamiltonian
        self.qubit_hamiltonian = tmp_hamiltonian

        return expectation

    def get_rdm(self, var_params, resample=False, sum_spin=True, ref_state=Circuit()):
        """Compute the 1- and 2- RDM matrices using the VQE energy evaluation.
        This method allows to combine the DMET problem decomposition technique
        with the VQE as an electronic structure solver. The RDMs are computed by
        using each fermionic Hamiltonian term, transforming them and computing
        the elements one-by-one. Note that the Hamiltonian coefficients will not
        be multiplied as in the energy evaluation. The first element of the
        Hamiltonian is the nuclear repulsion energy term, not the Hamiltonian
        term.

        Args:
            var_params (numpy.array or list): variational parameters to use for
                rdm calculation
            resample (bool): Whether to resample saved frequencies. get_rdm with
                savefrequencies=True must be called or a dictionary for each
                qubit terms' frequencies must be set to self.rdm_freq_dict
            sum_spin (bool): If True, the spin-summed 1-RDM and 2-RDM will be
                returned. If False, the full 1-RDM and 2-RDM will be returned.
            ref_state (Circuit): A reference state preparation circuit.

        Returns:
            (numpy.array, numpy.array): One & two-particle spin summed RDMs if
                sumspin=True or the full One & two-Particle RDMs if
                sumspin=False.
        """

        self.ansatz.update_var_params(var_params)

        # Initialize the RDM arrays
        n_mol_orbitals = self.molecule.n_active_mos
        n_spin_orbitals = self.molecule.n_active_sos
        rdm1_spin = np.zeros((n_spin_orbitals,) * 2, dtype=complex)
        rdm2_spin = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

        # If resampling is requested, check that a previous savefrequencies run has been called
        if resample:
            if hasattr(self, "rdm_freq_dict"):
                qb_freq_dict = self.rdm_freq_dict
                resampled_expect_dict = dict()
            else:
                raise AttributeError("Need to run RDM calculation with savefrequencies=True")
        else:
            qb_freq_dict = dict()
            qb_expect_dict = dict()

        # Loop over each element of Hamiltonian (non-zero value)
        for key in self.molecule.fermionic_hamiltonian.terms:
            # Ignore constant / empty term
            if not key:
                continue

            # Assign indices depending on one- or two-body term
            length = len(key)
            if (length == 2):
                iele, jele = (int(ele[0]) for ele in tuple(key[0:2]))
            elif (length == 4):
                iele, jele, kele, lele = (int(ele[0]) for ele in tuple(key[0:4]))

            # Create the Hamiltonian with the correct key (Set coefficient to one)
            hamiltonian_temp = FermionOperator(key)

            # Obtain qubit Hamiltonian
            qubit_hamiltonian2 = fermion_to_qubit_mapping(fermion_operator=hamiltonian_temp,
                                                          mapping=self.qubit_mapping,
                                                          n_spinorbitals=self.molecule.n_active_sos,
                                                          n_electrons=self.molecule.n_active_electrons,
                                                          up_then_down=self.up_then_down,
                                                          spin=self.molecule.spin)
            qubit_hamiltonian2.compress()

            # Run through each qubit term separately, use previously calculated result for the qubit term or
            # calculate and save results for that qubit term
            opt_energy2 = 0.
            for qb_term, qb_coef in qubit_hamiltonian2.terms.items():
                if qb_term:
                    if qb_term not in qb_freq_dict:
                        if resample:
                            warnings.warn(f"Warning: rerunning circuit for missing qubit term {qb_term}")
                        basis_circuit = Circuit(measurement_basis_gates(qb_term))
                        full_circuit = ref_state + self.ansatz.circuit + basis_circuit
                        qb_freq_dict[qb_term], _ = self.backend.simulate(full_circuit)
                    if resample:
                        if qb_term not in resampled_expect_dict:
                            resampled_freq_dict = get_resampled_frequencies(qb_freq_dict[qb_term], self.backend.n_shots)
                            resampled_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, resampled_freq_dict)
                        expectation = resampled_expect_dict[qb_term]
                    else:
                        if qb_term not in qb_expect_dict:
                            qb_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, qb_freq_dict[qb_term])
                        expectation = qb_expect_dict[qb_term]
                    opt_energy2 += qb_coef * expectation
                else:
                    opt_energy2 += qb_coef

            # Put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_spin[iele, jele] += opt_energy2
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] += opt_energy2

        # save rdm frequency dictionary
        self.rdm_freq_dict = qb_freq_dict

        if sum_spin:
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # Construct spin-summed 1-RDM
            for i, j in itertools.product(range(n_spin_orbitals), repeat=2):
                rdm1_np[i//2, j//2] += rdm1_spin[i, j]

            # Construct spin-summed 2-RDM
            for i, j, k, l in itertools.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[i//2, j//2, k//2, l//2] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin

    def get_rdm_uhf(self, var_params, resample=False, ref_state=Circuit()):
        """Compute the 1- and 2- RDM matrices using the VQE energy evaluation.
        This method allows to combine the DMET problem decomposition technique
        with the VQE as an electronic structure solver. The RDMs are computed by
        using each fermionic Hamiltonian term, transforming them and computing
        the elements one-by-one. Note that the Hamiltonian coefficients will not
        be multiplied as in the energy evaluation. The first element of the
        Hamiltonian is the nuclear repulsion energy term, not the Hamiltonian
        term.

        Args:
            var_params (numpy.array or list): variational parameters to use for
                rdm calculation
            resample (bool): Whether to resample saved frequencies. get_rdm with
                savefrequencies=True must be called or a dictionary for each
                qubit terms' frequencies must be set to self.rdm_freq_dict
            ref_state (Circuit): A reference state preparation circuit.

        Returns: TODO
            (numpy.array, numpy.array): One & two-particle spin summed RDMs if
                sumspin=True or the full One & two-Particle RDMs if
                sumspin=False.
        """

        self.ansatz.update_var_params(var_params)

        # Initialize the RDM arrays
        n_mol_orbitals = max(self.molecule.n_active_mos)
        rdm1_np_a = np.zeros((n_mol_orbitals,) * 2)
        rdm1_np_b = np.zeros((n_mol_orbitals,) * 2)
        rdm2_np_a = np.zeros((n_mol_orbitals,) * 4)
        rdm2_np_b = np.zeros((n_mol_orbitals,) * 4)
        rdm2_np_ba = np.zeros((n_mol_orbitals,) * 4)

        # If resampling is requested, check that a previous savefrequencies run has been called
        if resample:
            if hasattr(self, "rdm_freq_dict"):
                qb_freq_dict = self.rdm_freq_dict
                resampled_expect_dict = dict()
            else:
                raise AttributeError("Need to run RDM calculation with savefrequencies=True")
        else:
            qb_freq_dict = dict()
            qb_expect_dict = dict()

        # Loop over each element of Hamiltonian (non-zero value)
        for key in self.molecule.fermionic_hamiltonian.terms:
            # Ignore constant / empty term
            if not key:
                continue

            # Assign indices depending on one- or two-body term
            length = len(key)
            # One-body terms.
            if(length == 2):
                pele, qele = int(key[0][0]), int(key[1][0])
                iele, jele = pele // 2, qele // 2
                iele_r, jele_r = pele % 2, qele % 2
            # Two-body terms.
            elif(length == 4):
                pele, qele, rele, sele = int(key[0][0]), int(key[1][0]), int(key[2][0]), int(key[3][0])
                iele, jele, kele, lele = pele // 2, qele // 2, rele // 2, sele // 2
                iele_r, jele_r, kele_r, lele_r = pele % 2, qele % 2, rele % 2, sele % 2

            # Create the Hamiltonian with the correct key (Set coefficient to one)
            hamiltonian_temp = FermionOperator(key)

            # Obtain qubit Hamiltonian
            qubit_hamiltonian2 = fermion_to_qubit_mapping(fermion_operator=hamiltonian_temp,
                                                          mapping=self.qubit_mapping,
                                                          n_spinorbitals=self.molecule.n_active_sos,
                                                          n_electrons=self.molecule.n_active_electrons,
                                                          up_then_down=self.up_then_down,
                                                          spin=self.molecule.spin)
            qubit_hamiltonian2.compress()

            # Run through each qubit term separately, use previously calculated result for the qubit term or
            # calculate and save results for that qubit term
            opt_energy2 = 0.
            for qb_term, qb_coef in qubit_hamiltonian2.terms.items():
                if qb_term:
                    if qb_term not in qb_freq_dict:
                        if resample:
                            warnings.warn(f"Warning: rerunning circuit for missing qubit term {qb_term}")
                        basis_circuit = Circuit(measurement_basis_gates(qb_term))
                        full_circuit = ref_state + self.ansatz.circuit + basis_circuit
                        qb_freq_dict[qb_term], _ = self.backend.simulate(full_circuit)
                    if resample:
                        if qb_term not in resampled_expect_dict:
                            resampled_freq_dict = get_resampled_frequencies(qb_freq_dict[qb_term], self.backend.n_shots)
                            resampled_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, resampled_freq_dict)
                        expectation = resampled_expect_dict[qb_term]
                    else:
                        if qb_term not in qb_expect_dict:
                            qb_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, qb_freq_dict[qb_term])
                        expectation = qb_expect_dict[qb_term]
                    opt_energy2 += qb_coef * expectation
                else:
                    opt_energy2 += qb_coef

            # Put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                if (iele_r, jele_r) == (0, 0):
                    rdm1_np_a[iele, jele] += opt_energy2
                elif (iele_r, jele_r) == (1, 1):
                    rdm1_np_b[iele, jele] += opt_energy2
            elif length == 4:
                if ((iele != lele) or (jele != kele)):
                    if (iele_r, jele_r, kele_r, lele_r) == (0, 0, 0, 0):
                        rdm2_np_a[lele, iele, kele, jele] += 0.5 * opt_energy2
                        rdm2_np_a[iele, lele, jele, kele] += 0.5 * opt_energy2
                    elif (iele_r, jele_r, kele_r, lele_r) == (1, 1, 1, 1):
                        rdm2_np_b[lele, iele, kele, jele] += 0.5 * opt_energy2
                        rdm2_np_b[iele, lele, jele, kele] += 0.5 * opt_energy2
                    elif (iele_r, jele_r, kele_r, lele_r) == (0, 1, 1, 0):
                        rdm2_np_ba[iele, lele, jele, kele] += 0.5 * opt_energy2
                        rdm2_np_ba[lele, iele, kele, jele] += 0.5 * opt_energy2
                else:
                    if (iele_r, jele_r, kele_r, lele_r) == (0, 0, 0, 0):
                        rdm2_np_a[iele, lele, jele, kele] += opt_energy2
                    elif (iele_r, jele_r, kele_r, lele_r) == (1, 1, 1, 1):
                        rdm2_np_b[iele, lele, jele, kele] += opt_energy2
                    elif (iele_r, jele_r, kele_r, lele_r) == (0, 1, 1, 0):
                        rdm2_np_ba[iele, lele, jele, kele] += opt_energy2

        # save rdm frequency dictionary
        self.rdm_freq_dict = qb_freq_dict

        return (rdm1_np_a, rdm1_np_b), (rdm2_np_a, rdm2_np_ba, rdm2_np_b)

    def _default_optimizer(self, func, var_params):
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
