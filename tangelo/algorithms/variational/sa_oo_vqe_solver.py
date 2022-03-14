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

"""Module that defines the SA-OO-VQE algorithm

Ref:

"""
from copy import copy

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from tangelo.algorithms.variational.sa_vqe_solver import SA_VQESolver, BuiltInAnsatze


class SA_OO_Solver:
    """State Averaged Orbital Optimized Solver class. This is an iterative algorithm that uses VQE. Methods
    are defined to rank operators with respect to their influence on the total
    energy.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        ref_states (list): List of vectors defining the reference state occupations used for the system.
        tol (float): Maximum energy difference before convergence
        max_cycles (int): Maximum number of iterations for sa-oo-vqe
        qubit_mapping (str): One of the supported qubit mapping identifiers.
        up_then_down (bool): Spin orbitals ordering.
        optimizer (func): Optimization function for VQE minimization.
        backend_options (dict): Backend options for the underlying VQE object.
        verbose (bool): Flag for verbosity of VQE.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "ref_states": None,
                           "tol": 1e-3,
                           "max_cycles": 15,
                           "qubit_mapping": "jw",
                           "up_then_down": False,
                           "optimizer": self.LBFGSB_optimizer,
                           "backend_options": default_backend_options,
                           "verbose": False,
                           "ansatz": BuiltInAnsatze.UCCGD}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                # TODO Raise a warning instead, that variable will not be used unless user made mods to code
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if self.molecule is None:
            raise ValueError(f"A molecule object must be provided when instantiating {self.__class__.__name__}.")

        if self.ref_states is None:
            raise ValueError(f"spins are required to determine the state")

        self.converged = False
        self.iteration = 0
        self.energies = list()

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None

    def build(self):
        """Builds the underlying objects required to run the SA-OO-VQE
        algorithm.
        """

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles.
        self.vqe_options = {"molecule": self.molecule,
                            "ref_states": self.ref_states,
                            "optimizer": self.optimizer,
                            "backend_options": self.backend_options,
                            "ansatz": self.ansatz,
                            "qubit_mapping": self.qubit_mapping,
                            "up_then_down": self.up_then_down
                            }

        self.sa_vqe_solver = SA_VQESolver(self.vqe_options)
        self.sa_vqe_solver.build()

    def simulate(self):
        """Performs the ADAPT cycles. Each iteration, a VQE minimization is
        done.
        """
        # run initial sa_vqe
        for iter in range(self.max_cycles):
            energy_vqe = self.sa_vqe_solver.simulate()
            self.energies.append(self.energy_from_rdms())
            print(energy_vqe, self.energies[-1])
            if iter > 0 and abs(self.energies[-1]-self.energies[-2]) < self.tol:
                break
            u_mat = self.generate_oo_unitary()
            self.sa_vqe_solver.molecule.mean_field.mo_coeff = self.sa_vqe_solver.molecule.mean_field.mo_coeff @ u_mat
            print(self.energy_from_rdms())
            self.sa_vqe_solver.build()
            # self.vqe_solver.initial_var_params = copy(self.vqe_solver.optimal_var_params)

    def LBFGSB_optimizer(self, func, var_params):
        """Default optimizer for ADAPT-VQE."""

        result = minimize(func, var_params, method="L-BFGS-B",
                          options={"disp": False, "maxiter": 100, "gtol": 1e-10, "iprint": -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun

        # Reconstructing the optimal circuit at the end of the ADAPT iterations
        # or when the algorithm has converged.
        if self.converged or self.iteration == self.max_cycles:
            self.ansatz.build_circuit(self.optimal_var_params)
            self.optimal_circuit = self.sa_vqe_solver.ansatz.circuit

        if self.verbose:
            print(f"VQESolver optimization results:")
            print(f"\tOptimal VQE energy: {result.fun}")
            print(f"\tOptimal VQE variational parameters: {result.x}")
            print(f"\tNumber of Iterations : {result.nit}")
            print(f"\tNumber of Function Evaluations : {result.nfev}")
            print(f"\tNumber of Gradient Evaluations : {result.njev}")

        return result.fun, result.x

    def energy_from_rdms(self):
        "Calculate energy from rdms generated from SA_VQESolver"
        fcore, foneint, ftwoint = self.sa_vqe_solver.molecule.get_full_space_integrals()
        ftwoint = ftwoint.transpose(0, 3, 1, 2)
        occupied_indices = self.molecule.frozen_occupied
        active_indices = self.molecule.active_mos
        # Determine core constant
        core_constant = 0
        for i in occupied_indices:
            core_constant += 2 * foneint[i, i]
            for j in occupied_indices:
                core_constant += (2 * ftwoint[i, i, j, j] -
                                  ftwoint[i, j, i, j])

        active_energy = 0
        v_mat = np.zeros((foneint.shape[0], foneint.shape[0]))
        for t in active_indices:
            for u in active_indices:
                for i in occupied_indices:
                    v_mat[u, t] += (
                            2 * ftwoint[i, i, t, u] -
                            ftwoint[i, t, i, u])

        n_active_mos = self.molecule.n_active_mos
        one_rdm = np.zeros((n_active_mos, n_active_mos))
        two_rdm = np.zeros((n_active_mos, n_active_mos, n_active_mos, n_active_mos))
        num_refs = len(self.ref_states)
        for i in range(num_refs):
            one_rdm += self.sa_vqe_solver.rdms[i][0].real/num_refs
            two_rdm += self.sa_vqe_solver.rdms[i][1].real/num_refs/2
        for ti, t in enumerate(active_indices):
            for ui, u in enumerate(active_indices):
                active_energy += one_rdm[ti, ui]*(foneint[t, u] + v_mat[t, u])
                for vi, v in enumerate(active_indices):
                    for wi, w in enumerate(active_indices):
                        active_energy += two_rdm[ti, ui, vi, wi]*ftwoint[t, u, v, w]

        return fcore+core_constant+active_energy

    def generate_oo_unitary(self):
        """Generate the orbital optimization unitary that rotates the orbitals. It uses a single Newton-Raphson step
        with the Hessian calculated analytically."""
        _, foneint, ftwoint = self.sa_vqe_solver.molecule.get_full_space_integrals()
        ftwoint = ftwoint.transpose(0, 3, 1, 2)
        occupied_indices = self.molecule.frozen_occupied
        unoccupied_indices = self.molecule.frozen_virtual
        active_indices = self.molecule.active_mos
        n_mos = self.molecule.n_mos
        n_active_mos = self.molecule.n_active_mos
        # Determine core constant

        one_rdm = np.zeros((n_active_mos, n_active_mos))
        two_rdm = np.zeros((n_active_mos, n_active_mos, n_active_mos, n_active_mos))
        num_refs = len(self.ref_states)
        for i in range(num_refs):
            one_rdm += self.sa_vqe_solver.rdms[i][0].real*self.sa_vqe_solver.weights[i]
            two_rdm += self.sa_vqe_solver.rdms[i][1].real*self.sa_vqe_solver.weights[i]/2

        f_mat = np.zeros((n_mos, n_mos))
        fi_mat = np.zeros((n_mos, n_mos))
        fa_mat = np.zeros((n_mos, n_mos))

        for p in range(n_mos):
            for q in range(n_mos):
                fi_mat[p, q] = foneint[p, q]
                for i in occupied_indices:
                    fi_mat[p, q] += 2*ftwoint[p, q, i, i]-ftwoint[p, i, q, i]
                for ti, t in enumerate(active_indices):
                    for ui, u in enumerate(active_indices):
                        fa_mat[p, q] += one_rdm[ti, ui]*(ftwoint[p, q, t, u]-1/2*ftwoint[p, t, q, u])

        for i in occupied_indices:
            for q in range(n_mos):
                f_mat[i, q] = 2*(fa_mat[i, q]+fi_mat[i, q])

        for ti, t in enumerate(active_indices):
            for ui, u in enumerate(active_indices):
                for q in range(n_mos):
                    f_mat[t, q] += one_rdm[ti, ui]*fi_mat[q, u]
                    for vi, v in enumerate(active_indices):
                        for xi, x in enumerate(active_indices):
                            f_mat[t, q] += 2*two_rdm[ti, ui, vi, xi]*ftwoint[q, u, v, x]

        d2ed2x = np.zeros((n_mos, n_mos, n_mos, n_mos))
        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for j in occupied_indices:
                    for ui, u in enumerate(active_indices):
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[i, t, j, u] += 2*(two_rdm[ui, ti, vi, xi]*ftwoint[v, x, i, j]+(two_rdm[ui, xi, vi, ti] +
                                                         two_rdm[ui, xi, ti, vi])*ftwoint[v, i, x, j])
                            d2ed2x[i, t, j, u] += ((int(t == v)-one_rdm[ti, vi])*(4*ftwoint[v, i, u, j]-ftwoint[u, i, v, j]-ftwoint[u, v, i, j]) +
                                                   (int(u == v)-one_rdm[ui, vi])*(4*ftwoint[v, j, t, i]-ftwoint[t, j, v, i]-ftwoint[t, v, i, j]))
                        d2ed2x[i, t, j, u] += (one_rdm[ti, ui]*fi_mat[i, j]+int(i == j)*(2*fi_mat[t, u]+2*fa_mat[t, u] -
                                               f_mat[t, u])-2*int(t == u)*(fi_mat[i, j]+fa_mat[i, j]))

        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for j in occupied_indices:
                    for a in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            d2ed2x[i, t, j, a] += (2*int(t == v)-one_rdm[ti, vi])*(4*ftwoint[a, j, v, i]-ftwoint[a, v, i, j]-ftwoint[a, i, v, j])
                        d2ed2x[i, t, j, a] += 2*int(i == j)*(fi_mat[a, t]+fa_mat[a, t])-1/2*int(i == j)*f_mat[t, a]

        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for ui, u in enumerate(active_indices):
                    for a in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[i, t, u, a] += (-2)*(two_rdm[ti, ui, vi, xi]*ftwoint[a, i, v, x]+(two_rdm[ti, vi, ui, xi] +
                                                            two_rdm[ti, vi, xi, ui])*ftwoint[a, x, v, i])
                            d2ed2x[i, t, u, a] += one_rdm[ui, vi]*(4*ftwoint[a, v, t, i]-ftwoint[a, i, t, v]-ftwoint[a, t, v, i])
                        d2ed2x[i, t, u, a] += (-1)*one_rdm[ti, ui]*fi_mat[a, i]+int(t == u)*(fi_mat[a, i]+fa_mat[a, i])

        for i in occupied_indices:
            for a in unoccupied_indices:
                for j in occupied_indices:
                    for b in unoccupied_indices:
                        d2ed2x[i, a, j, b] += (2*(4*ftwoint[a, i, b, j]-ftwoint[a, b, i, j]-ftwoint[a, j, b, i]) +
                                               2*int(i == j)*(fi_mat[a, b]+fa_mat[a, b]) - 2*int(a == b)*(fi_mat[i, j]+fa_mat[i, j]))

        for i in occupied_indices:
            for a in unoccupied_indices:
                for ti, t in enumerate(active_indices):
                    for b in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            d2ed2x[i, a, t, b] += one_rdm[ti, vi]*(4*ftwoint[a, i, b, v]-ftwoint[a, v, b, i]-ftwoint[a, b, v, i])
                        d2ed2x[i, a, t, b] += (-1)*int(a == b)*(fi_mat[t, i]+fa_mat[t, i]) - 1/2*int(a == b)*f_mat[t, i]

        for ti, t in enumerate(active_indices):
            for a in unoccupied_indices:
                for ui, u in enumerate(active_indices):
                    for b in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[t, a, u, b] += 2*(two_rdm[ti, ui, vi, xi]*ftwoint[a, b, v, x]+(two_rdm[ti, xi, vi, ui] +
                                                         two_rdm[ti, xi, ui, vi])*ftwoint[a, x, b, v])
                                for yi, y in enumerate(active_indices):
                                    d2ed2x[t, a, u, b] -= (int(a == b)*(two_rdm[ti, vi, xi, yi]*ftwoint[u, v, x, y] +
                                                           two_rdm[ui, vi, xi, yi]*ftwoint[t, v, x, y]))
                            d2ed2x[t, a, u, b] += (-1/2)*int(a == b)*(one_rdm[ti, vi]*fi_mat[u, v]+one_rdm[ui, vi]*fi_mat[t, v])
                        d2ed2x[t, a, u, b] += one_rdm[ti, ui]*fi_mat[a, b]
                        # d2ed2x[t, a, u, b] += one_rdm[ti, ui]*fi_mat[a, b] - int(a == b)*f_mat[t, u]

        ivals = occupied_indices + active_indices
        jvals = active_indices + unoccupied_indices
        n_params = 0
        for i in ivals:
            for j in jvals:
                if (j > i and not (i in active_indices and j in active_indices)):
                    n_params += 1
        hess = np.zeros((n_params, n_params))
        dedx = np.zeros(n_params)
        p1 = -1
        for i in ivals:
            for j in jvals:
                if (j > i and not (i in active_indices and j in active_indices)):
                    p1 += 1
                    dedx[p1] = 2*(f_mat[i, j]-f_mat[j, i])
                    p2 = -1
                    for k in ivals:
                        for ll in jvals:
                            if (ll > k and not (k in active_indices and ll in active_indices)):
                                p2 += 1
                                hess[p1, p2] = d2ed2x[i, j, k, ll]*2
                                hess[p2, p1] = d2ed2x[i, j, k, ll]*2

        E, _ = np.linalg.eigh(hess)
        fac = abs(E[0])*2 if E[0] < 0 else 0
        hess = hess + np.eye(n_params)*fac
        knew = -np.linalg.solve(hess, dedx)

        mat_rep = np.zeros((n_mos,  n_mos))
        p1 = -1
        for i in ivals:
            for j in jvals:
                if (j > i and not (i in active_indices and j in active_indices)):
                    p1 += 1
                    mat_rep[i, j] = knew[p1]
                    mat_rep[j, i] = -knew[p1]

        return expm(-mat_rep)
