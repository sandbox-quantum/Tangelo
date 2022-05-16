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

"""

Refs:
    1. I. G. Ryabinkin, R. A. Lang, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.
"""


from openfermion import commutator

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.algorithms.variational.vqe_solver import VQESolver
from tangelo.toolboxes.ansatz_generator._qubit_cc import qcc_op_dress, qcc_op_compress


class iQCCSolver:
    """

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
        mapping (str): One of the supported qubit mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False.
        initial_var_params (str or array-like) : initial value for the classical
            optimizer.
        backend_options (dict) : parameters to build the tangelo.linq Simulator
            class.
        penalty_terms (dict): parameters for penalty terms to append to target
            qubit Hamiltonian (see penaly_terms for more details).
        ansatz_options (dict): parameters for the given ansatz (see given ansatz
            file for details).
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        deqcc_thresh (float): threshold for the difference in iQCC energies between
            consecutive iterations required for convergence of the algorithm.
            Default, 1e-6 Hartree.
        max_iqcc_iter (int): maximum number of iQCC iterations allowed before termination.
            Default, 100.
        max_iqcc_retries (int): if the iQCC energy for a given iteration is not lower than
            the value from the previous iteration, the iQCC parameters are reinitialized
            and the VQE procedure will be attempted up to max_iqcc_retries times. If unsuccessful
            after max_iqcc_retries attempts, the iQCC parameters are all set to 0 and the QMF
            Bloch angles from the previous iteration are used. Default, 10.
        compress_qubit_ham (bool): controls whether the qubit Hamiltonian is compressed
            after dressing with the current set of generators at the end of each iQCC iteration.
            Default, False.
        compress_epsilon (float): parameter required for compressing intermediate iQCC Hamiltonians
            using the Froebenius norm. Discarding terms in this manner will not alter the
            eigenspeectrum of intermediate Hamiltonians by more than compress_epsilon.
            Default, 1.59e-3 Hartree.
        verbose (bool): Flag for iQCC-VQE verbosity. Default, False.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "mapping": "jw",
                           "up_then_down": False,
                           "initial_var_params": None,
                           "backend_options": default_backend_options,
                           "penalty_terms": None,
                           "ansatz_options": None,
                           "qubit_hamiltonian": None,
                           "deqcc_thresh": 1e-6,
                           "max_iqcc_iter": 100,
                           "max_iqcc_retries": 10,
                           "compress_qubit_ham": False,
                           "compress_epsilon": 1.59e-3,
                           "verbose": False}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                raise KeyError(f"Keyword :: {k}, not available in iQCCSolver")

        if not self.molecule:
            raise ValueError("An instance of SecondQuantizedMolecule is required for initializing iQCCSolver.")

        self.converged = False
        self.iteration = 0
        self.energies = list()
        self.circuits = list()
        self.resources = list()
        self.generators = list()
        self.amplitudes = list()
        self.n_qubit_ham_terms = list()

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None

    def build(self):
        """Builds the underlying objects required to run the iQCC-VQE algorithm."""

        if ""
        self.qcc_ansatz = QCC(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
        self.qcc_ansatz.build()

        # Build an instance of VQESolver with options that remain fixed during the iQCC-VQE process.
        self.vqe_solver_options = {"molecule": self.molecule,
                           "qubit_mapping": self.qubit_mapping,
                           "ansatz": qcc_ansatz, 
                           "initial_var_params": self.initial_var_params,
                           "backend_options": self.backend_options,
                           "penalty_terms": self.penalty_terms,
                           "up_then_down": self.up_then_down,
                           "qubit_hamiltonian": self.qubit_hamiltonian,
                           "verbose": self.verbose}

        self.vqe_solver = VQESolver(self.vqe_solver_options)
        self.vqe_solver.build()

    def simulate(self):
        """Performs iQCC-VQE cycles. During each iteration, a VQE minimization is
        performed with the current set of QCC Pauli word generators and corresponding
        amplitudes.
        """

        # initialize quantities; initialize eqcc_old as the reference mean-field energy
        sim, e_qcc, delta_eqcc = Simulator(), 0., self.deqcc_thresh 
        qmf_qubit_ham, qmf_circuit = self.qmf_ansatz.qubit_ham, qmf_ansatz.circuit
        eqcc_old = sim.get_expectation_value(qmf_qubit_ham, qmf_circuit)
        n_gen = len(self.qcc_ansatz.qcc_op_list)
        while abs(delta_eqcc) >= self.deqcc_thresh and self.iteration < self.max_iqcc_iter:
            # check for at least one Pauli word generator and amplitude
            if self.qcc_ansatz.qcc_op_list and self.qcc_ansatz.var_params:
                # get the QCC energy at the current iter & compute difference with previous iter
                e_qcc = self.vqe_solver.simulate()
                delta_eqcc = e_qcc - eqcc_old
            # if there are none, then terminate the iQCC loop
            else:
                delta_eqcc = 0.
            if delta_eqcc < 0.:
                eqcc_old = e_qcc
                self._update_iqcc_solver()
                self.iteration += 1
                self.qcc_ansatz.qubit_ham = qcc_op_dress(self.qcc_ansatz.qubit_ham,
                                                         self.qcc_ansatz.var_params,
                                                         self.qcc_ansatz.qcc_op_list)
                if self.compress_qubit_ham:
                    self.qcc_ansatz.qubit_ham = qcc_op_compress(self.qcc_ansatz.qubit_ham,
                                                                self.qcc_ansatz.n_qubits,
                                                                self.compress_epsilon)
                                                  
            else:
                n_retry = 0
                while delta_eqcc >= 0.0 and n_retry < self.max_iqcc_retries:
                    for i in range(n_gen):
                        qcc_taus.append(random_uniform(-0.1, 0.1))
                    e_qcc = self.vqe_solver.simulate()
                    delta_eqcc = e_qcc - eqcc_old
                    n_retry += 1
                if delta_eqcc < 0.:
                else:
                    self.qcc_ansatz.set_var_params([0.] * n_gen)
        return self.energies[-1]

    def get_resources(self):
        """Returns a dictionary containing the optimal QCC energy, set of Pauli word
        generators, amplitudes, circuit, number of qubit Hamiltonian terms, and quantum
        resource estimations at each iteration of the iQCC-VQE solver."""

        iqcc_resources = dict()
        iqcc_resources["energies"] = self.energies
        iqcc_resources["circuits"] = self.circuits
        iqcc_resources["resources"] = self.resources
        iqcc_resources["generators"] = self.generators
        iqcc_resources["amplitudes"] = self.amplitudes
        iqcc_resources["n_qham_terms"] = self.n_qubit_ham_terms
        return iqcc_resources

    def _update_iqcc_solver(self):
        """ Update the lists for the optimal QCC energy, set of Pauli word
        generators, amplitudes, circuit, number of qubit Hamiltonian terms, and quantum
        resource estimation at each iteration of the iQCC-VQE solver."""

        self.qcc_ansatz.var_params = self.vqe_solver.optimal_var_params
        self.qcc_ansatz.circuit = self.vqe_solver.optimal_circuit

        self.energies.append(self.vqe_solver.optimal_energy)
        self.circuits.append(self.qcc_ansatz.circuit)
        self.resources.append(self.vqe_solver.get_resources())
        self.generators.append(self.qcc_ansatz.qcc_op_list)
        self.amplitudes.append(self.qcc_ansatz.var_params)
        self.n_qubit_ham_terms.append(len(self.qcc_ansatz.qubit_ham.terms))

        self.qcc_ansatz.qubit_ham = qcc_op_dress(self.qcc_ansatz.qubit_ham, self.qcc_ansatz.var_params,
                                                 self.qcc_ansatz.qcc_op_list)
        if self.compress_qubit_ham:
            self.qcc_ansatz.qubit_ham = qcc_op_compress(self.qcc_ansatz.qubit_ham, self.compress_epsilon,
                                                        self.qcc_ansatz.n_qubits)
