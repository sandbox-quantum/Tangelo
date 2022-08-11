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
This module implements the iQCC-ILC VQE procedure of Refs. 1-2. It is
an iterative and  variational approach that combines an ansatz defined
as an exponentiated involutory linear combination (ILC) of mutually
anticommuting Pauli word generators with the QCC ansatz. A small number
of iterations are performed with the ILC ansatz prior to a single energy
evaluation with the QCC ansatz. The advantage of this method over the
iQCC VQE procedure is that Hamiltonian dressing after each iteration
with the set of ILC generators results in quadratic growth of the
number of terms, which is an improvement over the exponential growth
that occurs when QCC generators are used.
Refs:
    1. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        arXiv:2002.05701v1, 2020, 1–10.
    2. R. A. Lang, I. G. Ryabinkin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2021, 17, 1, 66–78.
"""

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.ilc import ILC
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.algorithms.variational.vqe_solver import VQESolver
from tangelo.toolboxes.ansatz_generator._qubit_ilc import ilc_op_dress


class iQCC_ILC_solver:
    """The iQCC-ILC-VQE solver class combines the both the ILC and ILC ansatze
    Classes with the VQESolver class to perform an iterative and variational
    procedure to compute the total iQCC-ILC energy for a given Hamiltonian.
    The algorithm is outlined below:
    (1) For a user-specified number of iterations, compute the ILC energy:
        (a) prepare/purify the QMF wave function, obtain the ACS of ILC
            generators, and initialize the ILC parameter set;
        (b) simulate the ILC energy through VQE minimization
        (c) dress the qubit Hamiltonian with the set of ILC generators and
            optimal parameters; optional: compress the dressed Hamiltonian
            via a technique using the Frobenius norm
    (2) With the ILC dressed Hamiltonian, obtain the DIS of QCC generators,
        and initialize QCC parameters
    (3) Perform a single VQE minimization of the QCC energy functional to
        obtain the final iQCC-ILC energy.
    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        qubit_mapping (str): One of the supported qubit mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False.
        initial_var_params (str or array-like): Initial values of the variational parameters
            for the classical optimizer.
        backend_options (dict): Parameters to build the tangelo.linq Simulator
            class.
        penalty_terms (dict): Parameters for penalty terms to append to target
            qubit Hamiltonian (see penaly_terms for more details).
        ilc_ansatz_options (dict): Parameters for ILC ansatz (see ILC ansatz
            file for details).
        qcc_ansatz_options (dict): Parameters for QCC ansatz (see QCC ansatz
            file for details).
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        max_ilc_iter (int): maximum number of ILC iterations allowed before termination.
            Default, 3.
        compress_qubit_ham (bool): controls whether the qubit Hamiltonian is compressed
            after dressing with the current set of generators at the end of each ILC iteration.
            Default, False.
        compress_eps (float): parameter required for compressing intermediate ILC Hamiltonians
            using the Froebenius norm. Discarding terms in this manner will not alter the
            eigenspectrum of intermediate Hamiltonians by more than compress_eps.
            Default, 1.59e-3 Hartree.
        verbose (bool): Flag for verbosity. Default, False.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "qubit_mapping": "jw",
                           "up_then_down": False,
                           "initial_var_params": None,
                           "backend_options": default_backend_options,
                           "penalty_terms": None,
                           "ilc_ansatz_options": dict(),
                           "qcc_ansatz_options": dict(),
                           "qubit_hamiltonian": None,
                           "max_ilc_iter": 3,
                           "compress_qubit_ham": False,
                           "compress_eps": 1.59e-3,
                           "verbose": False}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for param, val in opt_dict.items():
            if param in default_options:
                setattr(self, param, val)
            else:
                raise KeyError(f"The keyword {param} is not available in self.__class__.__name__.")

        if not self.molecule and not self.qubit_hamiltonian:
            raise ValueError("An instance of SecondQuantizedMolecule or QubitOperator "
                             "is required for initializing self.__class__.__name__.")

        # initialize variables and lists to store useful data from each ILC-VQE iteration
        self.energies = []
        self.iteration = 0
        self.terminate_ilc = False
        self.qmf_energy = None
        self.ilc_ansatz = None
        self.qcc_ansatz = None
        self.vqe_solver = None
        self.vqe_solver_options = None
        self.final_optimal_energy = None
        self.final_optimal_qmf_params = None
        self.final_optimal_ilc_params = None
        self.final_optimal_qcc_params = None

    def build(self):
        """Builds the underlying objects required to run the ILC-VQE algorithm."""

        # instantiate the ILC ansatz but do not build it here because vqe_solver builds it
        self.ilc_ansatz = ILC(self.molecule, self.qubit_mapping, self.up_then_down, **self.ilc_ansatz_options)

        # build an instance of VQESolver with options that remain fixed during the ILC-VQE routine
        self.vqe_solver_options = {"qubit_hamiltonian": self.ilc_ansatz.qubit_ham,
                                   "qubit_mapping": self.qubit_mapping,
                                   "ansatz": self.ilc_ansatz,
                                   "initial_var_params": self.initial_var_params,
                                   "backend_options": self.backend_options,
                                   "penalty_terms": self.penalty_terms,
                                   "up_then_down": self.up_then_down,
                                   "verbose": self.verbose}

        self.vqe_solver = VQESolver(self.vqe_solver_options)
        self.vqe_solver.build()

    def simulate(self):
        """Executes the ILC-VQE algorithm. During each iteration, a ILC-VQE minimization
        is performed with the current set of generators, amplitudes, and qubit Hamiltonian."""

        # initialize quantities and compute the QMF energy
        sim = Simulator()
        self.qmf_energy = sim.get_expectation_value(self.ilc_ansatz.qubit_ham, self.ilc_ansatz.qmf_circuit)
        if self.verbose:
            print(f"The qubit mean field energy = {self.qmf_energy}")

        # perform self.max_ilc_iter ILC-VQE minimizations;
        e_ilc = 0.
        while not self.terminate_ilc:
            # check that the ACS has at least one ILC generator to use; if not, terminate
            if self.ilc_ansatz.acs and self.ilc_ansatz.var_params.any():
                e_ilc = self.vqe_solver.simulate()
            else:
                self.terminate_ilc = True
                if self.verbose:
                    print("Terminating the ILC-VQE solver: the ACS of ILC generators is empty.")
            # update ILC-VQE simulation data
            if not self.terminate_ilc:
                self._update_ilc_solver(e_ilc)

        # perform a single QCC-VQE minimization to obtain the final iQCC-ILC energy
        # need to rebuild VQE Solver for the QCC ansatz first
        self._build_qcc()

        # check that the DIS has at least one QCC generator to use
        if self.qcc_ansatz.dis and self.qcc_ansatz.var_params.any():
            e_iqcc_ilc = self.vqe_solver.simulate()
            self._update_qcc_solver(e_iqcc_ilc)
        else:
            if self.verbose:
                print("Terminating the iQCC-ILC solver without evaluating the "
                      "the final QCC energy because the DIS of QCC generators "
                      "is empty for the given Hamiltonian.")

        return self.energies[-1]

    def get_resources(self):
        """Returns the quantum resource estimates for the final
        ILC-QCC-VQE iteration."""

        return self.vqe_solver.get_resources()

    def _build_qcc(self):
        """Builds the underlying QCC objects required to run the second part of the
        iQCC-ILC-VQE algorithm."""

        # instantiate the QCC ansatz with the final ILC dressed Hamiltonian and optimal QMF params
        self.qcc_ansatz_options["qmf_var_params"] = self.final_optimal_qmf_params
        self.qcc_ansatz_options["qubit_ham"] = self.ilc_ansatz.qubit_ham
        self.qcc_ansatz = QCC(self.molecule, self.qubit_mapping, self.up_then_down, **self.qcc_ansatz_options)

        # build an instance of VQESolver with options that remain fixed during the ILC-VQE routine
        self.vqe_solver_options = {"qubit_mapping": self.qubit_mapping,
                                   "ansatz": self.qcc_ansatz,
                                   "initial_var_params": self.initial_var_params,
                                   "backend_options": self.backend_options,
                                   "penalty_terms": self.penalty_terms,
                                   "up_then_down": self.up_then_down,
                                   "qubit_hamiltonian": self.ilc_ansatz.qubit_ham,
                                   "verbose": self.verbose}

        self.vqe_solver = VQESolver(self.vqe_solver_options)
        self.vqe_solver.build()

    def _update_ilc_solver(self, e_ilc):
        """This function serves several purposes for the ILC-VQE solver
        part of the iQCC-ILC algorithm:

            (1) Updates the ILC energy, generators, QMF Bloch angles,
                ILC amplitudes, circuits, number of qubit Hamiltonian terms,
                and quantum resource estimates;
            (2) Dresses and compresses (optional) the qubit Hamiltonian
                with the generators and optimal amplitudes for the current
                iteration;
            (3) Prepares for the next iteration by rebuilding the DIS & ACS,
                reinitializing ILC parameters for the new set of generators,
                generating the circuit, and updating the classical optimizer.

        Args:
            e_ilc (float): the total ILC ansatz energy at the current iteration.
        """

        # get the optimal variational parameters and split them for qmf and ilc
        n_qubits = self.ilc_ansatz.n_qubits
        optimal_qmf_var_params = self.vqe_solver.optimal_var_params[:2*n_qubits]
        optimal_ilc_var_params = self.vqe_solver.optimal_var_params[2*n_qubits:]

        # update energy list and iteration number
        self.energies.append(e_ilc)
        self.iteration += 1

        if self.verbose:
            print(f"Iteration # {self.iteration}")
            print(f"ILC total energy = {e_ilc} Eh")
            print(f"ILC correlation energy = {e_ilc-self.qmf_energy} Eh")
            print(f"Optimal QMF variational parameters = {optimal_qmf_var_params}")
            print(f"Optimal ILC variational parameters = {optimal_ilc_var_params}")
            print(f"# of ILC generators = {len(self.ilc_ansatz.acs)}")
            print(f"ILC generators = {self.ilc_ansatz.acs}")
            print(f"ILC resource estimates = {self.get_resources()}")
            if self.iteration == 1:
                n_qham_terms = self.ilc_ansatz.qubit_ham.get_max_number_hamiltonian_terms(n_qubits)
                print(f"Maximum number of qubit Hamiltonian terms = {n_qham_terms}")

        # dress and (optionally) compress the qubit Hamiltonian for the next iteration
        self.ilc_ansatz.qubit_ham = ilc_op_dress(self.ilc_ansatz.qubit_ham, self.ilc_ansatz.acs,
                                                 optimal_ilc_var_params)
        if self.compress_qubit_ham:
            self.ilc_ansatz.qubit_ham.frobenius_norm_compression(self.compress_eps, n_qubits)

        # set dis, acs, and var_params to none to rebuild the dis & acs and initialize new parameters
        self.ilc_ansatz.dis = None
        self.ilc_ansatz.acs = None
        self.ilc_ansatz.var_params = None
        self.ilc_ansatz.build_circuit()

        self.vqe_solver.qubit_hamiltonian = self.ilc_ansatz.qubit_ham
        self.vqe_solver.initial_var_params = self.ilc_ansatz.var_params

        if self.iteration == self.max_ilc_iter:
            self.terminate_ilc = True
            self.final_optimal_qmf_params = optimal_qmf_var_params
            self.final_optimal_ilc_params = optimal_ilc_var_params
            if self.verbose:
                print(f"Terminating the ILC-VQE solver after {self.max_ilc_iter} iterations.")

    def _update_qcc_solver(self, e_iqcc_ilc):
        """Updates after the final QCC energy evaluation with the final ILC dressed
        Hamiltonian.

        Args:
            e_iqcc_ilc (float): the final iQCC-ILC ansatz energy.
        """

        # get the optimal variational parameters and split them for qmf and qcc ansatze
        n_qubits = self.qcc_ansatz.n_qubits
        self.final_optimal_qmf_var_params = self.vqe_solver.optimal_var_params[:2*n_qubits]
        self.final_optimal_qcc_var_params = self.vqe_solver.optimal_var_params[2*n_qubits:]

        # update energy list
        self.final_optimal_energy = e_iqcc_ilc
        self.energies.append(e_iqcc_ilc)

        if self.verbose:
            print("Final iQCC-ILC VQE simulation.")
            print(f"iQCC-ILC total energy = {e_iqcc_ilc} Eh")
            print(f"iQCC-ILC correlation energy = {e_iqcc_ilc-self.qmf_energy} Eh")
            print(f"Optimal QMF variational parameters = {self.final_optimal_qmf_var_params}")
            print(f"Optimal QCC variational parameters = {self.final_optimal_qcc_var_params}")
            print(f"# of QCC generators = {len(self.qcc_ansatz.dis)}")
            print(f"QCC generators = {self.qcc_ansatz.dis}")
            print(f"QCC resource estimates = {self.get_resources()}")
