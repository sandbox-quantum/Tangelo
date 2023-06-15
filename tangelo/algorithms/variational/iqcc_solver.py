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

"""
This module implements the iterative qubit coupled cluster (iQCC)-VQE
procedure of Ref. 1. It is a variational approach that utilizes the
the QCC ansatz to produce shallow circuits. The iterative procedure
allows a small number (1—10) of generators to be used for the QCC
This results in even shallower circuits and fewer quantum resources
for the iQCC approach relative to the native QCC method. A caveat
is that after each iteration, the qubit Hamiltonian is dressed with
the generators and optimal parameters, the result of which is an
exponential growth of the number of terms.

Refs:
    1. I. G. Ryabinkin, R. A. Lang, S. N. Genin, and A. F. Izmaylov.
        J. Chem. Theory Comput. 2020, 16, 2, 1055–1063.
"""

from typing import Union

from tangelo import SecondQuantizedMolecule
from tangelo.linq import get_backend
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.algorithms.variational.vqe_solver import VQESolver, BuiltInAnsatze
from tangelo.toolboxes.ansatz_generator._qubit_cc import qcc_op_dress


class iQCC_solver:
    """The iQCC-VQE solver class combines the QCC ansatz and VQESolver classes
    to perform an iterative and variational procedure to compute the total QCC
    energy for a given Hamiltonian. The algorithm is outlined below:

    (0) Prepare a qubit Hamiltonian, initialize QMF parameters, construct the
        DIS, select QCC generators, and initialize QCC amplitudes.
    (1) Simulate the QCC energy through VQE minimization.
    (2) Check if the energy is lowered relative to the previous iteration.
    (3) If the energy is lowered, proceed to (4); else, keep the QCC generators,
        re-initialize the amplitudes, and re-compute the energy. If after several
        attempts the energy is not lowered, set all QCC amplitudes to zero and
        use the QMF parameters from the previous iteration to compute the energy.
        This is guaranteed to yield a lower energy.
    (4) Check termination criteria: terminate if the change in energy is below a
        threshold, the DIS is empty, or the maximum number of iterations is reached.
    (5) If not terminated, dress the qubit Hamiltonian with the current QCC
        generators and optimal amplitudes.
    (6) Purify the QMF parameters, rebuild the DIS, and select generators for
        the next iteration; return to (1) and repeat until termination.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        qubit_mapping (str): One of the supported qubit mapping identifiers. Default, "jw".
        up_then_down (bool): Change basis ordering putting all spin up orbitals first,
            followed by all spin down. Default, False.
        initial_var_params (str or array-like): Initial values of the variational parameters
            for the classical optimizer.
        backend_options (dict): parameters to build the underlying compute backend (simulator, etc).
        penalty_terms (dict): Parameters for penalty terms to append to target
            qubit Hamiltonian (see penaly_terms for more details).
        ansatz_options (dict): Parameters for the chosen ansatz (see given ansatz
            file for details).
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        deqcc_thresh (float): threshold for the difference in iQCC energies between
            consecutive iterations required for convergence of the algorithm.
            Default, 1e-5 Hartree.
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
        compress_eps (float): parameter required for compressing intermediate iQCC Hamiltonians
            using the Froebenius norm. Discarding terms in this manner will not alter the
            eigenspeectrum of intermediate Hamiltonians by more than compress_eps.
            Default, 1.59e-3 Hartree.
        verbose (bool): Flag for verbosity. Default, False.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}

        copt_dict = opt_dict.copy()
        self.molecule: SecondQuantizedMolecule = copt_dict.pop("molecule", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.up_then_down: bool = copt_dict.pop("up_then_down", False)
        self.initial_var_params: Union[str, list] = copt_dict.pop("initial_var_params", None)
        self.backend_options: dict = copt_dict.pop("backend_options", default_backend_options)
        self.penalty_terms: dict = copt_dict.pop("penalty_terms", None)
        self.ansatz_options: dict = copt_dict.pop("ansatz_options", dict())
        self.qubit_hamiltonian: QubitOperator = copt_dict.pop("qubit_hamiltonian", None)
        self.deqcc_thresh: float = copt_dict.pop("deqcc_thresh", 1e-5)
        self.max_iqcc_iter: int = copt_dict.pop("max_iqcc_iter", 100)
        self.max_iqcc_retries: int = copt_dict.pop("max_iqcc_retries", 10)
        self.compress_qubit_ham: bool = copt_dict.pop("compress_qubit_ham", False)
        self.compress_eps: float = copt_dict.pop("compress_eps", 1.59e-3)
        self.verbose: bool = copt_dict.pop("verbose", False)

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        if not self.molecule:
            raise ValueError(f"An instance of SecondQuantizedMolecule is required for initializing {self.__class__.__name__}.")

        # initialize variables and lists to store useful data from each iQCC-VQE iteration
        self.energies = []
        self.iteration = 0
        self.converged = False
        self.qmf_energy = None
        self.qcc_ansatz = None
        self.vqe_solver = None
        self.vqe_solver_options = None
        self.final_optimal_energy = None
        self.final_optimal_qmf_params = None
        self.final_optimal_qcc_params = None

    def build(self):
        """Builds the underlying objects required to run the iQCC-VQE algorithm."""

        # build an instance of VQESolver with options that remain fixed during the iQCC-VQE routine
        self.vqe_solver_options = {"molecule": self.molecule,
                                   "qubit_mapping": self.qubit_mapping,
                                   "ansatz": BuiltInAnsatze.QCC,
                                   "initial_var_params": self.initial_var_params,
                                   "backend_options": self.backend_options,
                                   "penalty_terms": self.penalty_terms,
                                   "up_then_down": self.up_then_down,
                                   "qubit_hamiltonian": self.qubit_hamiltonian,
                                   "verbose": self.verbose,
                                   "ansatz_options": self.ansatz_options}

        self.vqe_solver = VQESolver(self.vqe_solver_options)
        self.vqe_solver.build()

        self.qcc_ansatz = self.vqe_solver.ansatz

    def simulate(self):
        """Executes the iQCC-VQE algorithm. During each iteration,
        QCC-VQE minimization is performed."""

        # initialize quantities; compute the QMF energy and set this as eqcc_old
        sim = get_backend()
        self.qmf_energy = sim.get_expectation_value(self.qcc_ansatz.qubit_ham, self.qcc_ansatz.qmf_circuit)
        e_qcc, eqcc_old, delta_eqcc = 0., self.qmf_energy, self.deqcc_thresh

        if self.verbose:
            print(f"The qubit mean field energy = {self.qmf_energy}")

        while not self.converged and self.iteration < self.max_iqcc_iter:
            # check that the DIS has at least one generator to use; otherwise terminate
            if self.qcc_ansatz.dis and self.qcc_ansatz.var_params.any():
                e_qcc = self.vqe_solver.simulate()
                delta_eqcc = e_qcc - eqcc_old
                eqcc_old = e_qcc
            else:
                self.converged = True
                if self.verbose:
                    print("Terminating the iQCC-VQE solver: the DIS of QCC generators is empty.")

            # check if unsuccessful: energy is not lowered and energy is not converged.
            if delta_eqcc > 0. and delta_eqcc >= self.deqcc_thresh:
                n_retry = 0
                if self.verbose:
                    print(f"The energy at iteration {self.iteration} is greater than the energy "
                          f"from the previous iteration. Making {self.max_iqcc_retries} attempts "
                          f"to find a lower energy solution")

                # make several attempts to obtain a lower energy
                while e_qcc > eqcc_old and n_retry < self.max_iqcc_retries:
                    self.qcc_ansatz.var_params = None
                    self.qcc_ansatz.update_var_params("random")
                    self.vqe_solver.initial_var_params = self.qcc_ansatz.var_params
                    e_qcc = self.vqe_solver.simulate()
                    n_retry += 1

                # check if energy was lowered; else zero the amplitudes and recompute
                if e_qcc < eqcc_old:
                    delta_eqcc = e_qcc - eqcc_old
                    eqcc_old = e_qcc
                else:
                    self.qcc_ansatz.var_params = None
                    self.qcc_ansatz.update_var_params("qmf_state")
                    self.vqe_solver.initial_var_params = self.qcc_ansatz.var_params
                    eqcc_old = e_qcc
                    e_qcc = self.vqe_solver.simulate()
                    delta_eqcc = e_qcc - eqcc_old

            # update simulation data and check convergence
            if not self.converged:
                self._update_iqcc_solver(delta_eqcc)

        return self.energies[-1]

    def get_resources(self):
        """Returns the quantum resource estimates for the final
        iQCC-VQE iteration."""

        return self.vqe_solver.get_resources()

    def _update_iqcc_solver(self, delta_eqcc):
        """This function serves several purposes after successful iQCC-VQE
        iterations:
            (1) updates/stores the energy, generators, QMF Bloch angles,
                QCC amplitudes, circuits, number of qubit Hamiltonian terms,
                and quantum resource estimates;
            (2) dresses/compresses the qubit Hamiltonian with the current
                generators and optimal amplitudes;
            (3) prepares for the next iteration by rebuilding the DIS,
                re-initializing the amplitudes for a new set of generators,
                generating the circuit, and updates the classical optimizer.
        """

        # get the optimal variational parameters and split them for qmf and qcc
        n_qubits = self.qcc_ansatz.n_qubits
        optimal_qmf_var_params = self.vqe_solver.optimal_var_params[:2*n_qubits]
        optimal_qcc_var_params = self.vqe_solver.optimal_var_params[2*n_qubits:]

        # update energy list and iteration number
        self.energies.append(self.vqe_solver.optimal_energy)
        self.iteration += 1

        if self.verbose:
            print(f"Iteration # {self.iteration}")
            print(f"iQCC total energy = {self.vqe_solver.optimal_energy} Eh")
            print(f"iQCC correlation energy = {self.vqe_solver.optimal_energy-self.qmf_energy} Eh")
            print(f"Optimal QMF variational parameters = {optimal_qmf_var_params}")
            print(f"Optimal QCC variational parameters = {optimal_qcc_var_params}")
            print(f"Number of iQCC generators = {len(self.qcc_ansatz.dis)}")
            print(f"iQCC generators = {self.qcc_ansatz.dis}")
            print(f"iQCC resource estimates = {self.get_resources()}")

        # Order of circuit does not follow dis list order, take order from pauli_to_angles_mapping
        n_qcc_params = self.qcc_ansatz.n_qcc_params
        ordered_dis, ordered_params = [None]*n_qcc_params, [None]*n_qcc_params
        for i, qu_gen in enumerate(self.qcc_ansatz.dis):
            pauli = next(iter(qu_gen.terms))
            # Hamiltonian must be transformed in reverse order of appearance in circuit
            pos = n_qcc_params - 1 - self.qcc_ansatz.pauli_to_angles_mapping[pauli]
            ordered_dis[pos] = qu_gen
            ordered_params[pos] = -optimal_qcc_var_params[i]

        self.qcc_ansatz.qubit_ham = qcc_op_dress(self.qcc_ansatz.qubit_ham, ordered_dis,
                                                 ordered_params)
        if self.compress_qubit_ham:
            self.qcc_ansatz.qubit_ham.frobenius_norm_compression(self.compress_eps, n_qubits)

        # set dis and var_params to none to rebuild the dis and initialize new amplitudes
        self.qcc_ansatz.dis = None
        self.qcc_ansatz.var_params = None
        self.qcc_ansatz.build_circuit()
        self.vqe_solver.initial_var_params = self.qcc_ansatz.var_params
        self.vqe_solver.qubit_hamiltonian = self.qcc_ansatz.qubit_ham

        if abs(delta_eqcc) < self.deqcc_thresh or self.iteration == self.max_iqcc_iter:
            self.converged = True
            self.final_optimal_energy = self.vqe_solver.optimal_energy
            self.final_optimal_qmf_params = optimal_qmf_var_params
            self.final_optimal_qcc_params = optimal_qcc_var_params

            if self.verbose:
                if abs(delta_eqcc) < self.deqcc_thresh:
                    print("Terminating the iQCC-VQE solver: energy convergence threshold achieved.")
                elif self.iteration == self.max_iqcc_iter:
                    print("Terminating the iQCC-VQE solver: maximum number of iQCC iterations reached.")
