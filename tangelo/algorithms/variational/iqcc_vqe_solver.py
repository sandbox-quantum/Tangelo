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

Ref:

"""

from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.toolboxes.ansatz_generator.qcc import QCC
from tangelo.toolboxes.ansatz_generator._qubit_cc import 
from tangelo.algorithms.variational.vqe_solver import VQESolver


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
        self.iteration = 1
        self.iqcc_energies = list()
        self.iqcc_generators = list()
        self.hqubit_terms = list()

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None

    def build(self):
        """Builds the underlying objects required to run the iQCC-VQE algorithm."""

        if ""
        self.qmf_ansatz = QMF(self.molecule, self.qubit_mapping, self.up_then_down)
        self.qmf_ansatz.build()

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
        """Performs iQCC-VQE cycles. Each iteration, a VQE minimization is
        done.
        """

        params = self.vqe_solver.ansatz.var_params
        e_qcc, eqcc_old, delta_eqcc = 0., 0., self.deqcc_thresh 

        while abs(delta_eqcc) >= self.deqcc_thresh and self.iteration < self.max_iqcc_iter:
            if :
                
                delta_eqcc = EQCC - eqcc_old
            else:
                delta_eqcc = 0.
            if (abs(delta_eqcc) >= deqcc_thresh and delta_eqcc < 0.0):
                eqcc_old = EQCC
                H_qubit, QMF_angles = iQCC_Update(H_qubit, N_qubit, S2_qubit, Sz_qubit, QMF_angles, QCC_gens, QCC_taus, NQb, NGen, EQMF, EQCC, delta_eqcc, iqcc_iter, scfdata)
                self.iteration += 1
            elif (abs(delta_eqcc) >= deqcc_thresh and delta_eqcc > 0.0):
                NGuess = 1
                while(abs(delta_eqcc) >= deqcc_thresh and delta_eqcc > 0.0 and NGuess <= QCC_max_guess):
                    EQCC_iter_old = eqcc_old
                    QCC_taus = list(Init_QMFState(scfdata))
                    for i in range(NGen):
                        QCC_taus.append(random_uniform(-0.1, 0.1))
                    EQCC, QCC_taus, QCC_success = QCC_Solver(H_qubit, QCC_gens, QCC_taus, NGen, NQb)
                    delta_eqcc = EQCC - EQCC_iter_old
                    NGuess += 1
                if (abs(delta_eqcc) >= deqcc_thresh and delta_eqcc < 0.0):
                    eqcc_old = EQCC
                    H_qubit, QMF_angles = iQCC_Update(H_qubit, N_qubit, S2_qubit, Sz_qubit, QMF_angles, QCC_gens, QCC_taus, NQb, NGen, EQMF, EQCC, delta_eqcc, iqcc_iter, scfdata)
                    iqcc_iter += 1
                elif (abs(delta_eqcc) >= deqcc_thresh and delta_eqcc > 0.0):
                    QCC_taus = list(QMF_angles)[:]
                    for i in range(NGen):
                        QCC_taus.append(0.0)

        # Construction of the ansatz. self.max_cycles terms are added, unless
        # all operator gradients are less than self.tol.
        while self.iteration < self.max_cycles:
            self.iteration += 1
            if self.verbose:
                print(f"Iteration {self.iteration} of ADAPT-VQE.")

            pool_select = self.rank_pool(self.pool_commutators, self.vqe_solver.ansatz.circuit,
                                         backend=self.vqe_solver.backend, tolerance=self.tol)

            # If pool selection returns an operator that changes the energy by
            # more than self.tol. Else, the loop is complete and the energy is
            # considered as converged.
            if pool_select > -1:

                # Adding a new operator + initializing its parameters to 0.
                # Previous parameters are kept as they were.
                params += [0.]
                if self.pool_type == 'fermion':
                    ielf.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select], self.fermionic_operators[pool_select])
                else:
                    self.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select])
                self.vqe_solver.initial_var_params = params

                # Performs a VQE simulation and append the energy to a list.
                # Also, forcing params to be a list to make it easier to append
                # new parameters. The behavior with a np.array is multiplication
                # with broadcasting (not wanted).
                self.vqe_solver.simulate()
                opt_energy = self.vqe_solver.optimal_energy
                params = list(self.vqe_solver.optimal_var_params)
                self.energies.append(opt_energy)
            else:
                self.converged = True
                break

        return self.energies[-1]

    def get_resources(self):
        """Returns an estimate of quantum resources required by the circuit at the current
        iQCC-VQE iteration."""

        return self.vqe_solver.get_resources()
