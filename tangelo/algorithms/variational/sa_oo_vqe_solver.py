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
[1] Saad Yalouz, Bruno Senjean, Jakob Gunther, Francesco Buda, Thomas E. O'Brien, Lucas Visscher, "A state-averaged
orbital-optimized hybrid quantum-classical algorithm for a democratic description of ground and excited states",
2021, Quantum Sci. Technol. 6 024004

"""

import numpy as np
from scipy.linalg import expm

from tangelo.algorithms.variational import SA_VQESolver


class SA_OO_Solver(SA_VQESolver):
    """State Averaged Orbital Optimized Solver class. This is an iterative algorithm that uses SA-VQE alternatively with an
    orbital optimization step.

    Users must first set the desired options of the SA_OO_Solver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the energy_estimation, simulate, get_rdm, or iterate methods. In particular, iterate
    runs the SA-OO algorithm, alternating calls to the SA_VQESolver and orbital optimization.

    Attributes:
        tol (float): Maximum energy difference before convergence
        max_cycles (int): Maximum number of iterations for sa-oo-vqe
        n_oo_per_iter (int): Number of orbital optimization Newton-Raphson steps per SA-OO-VQE iteration
        molecule (SecondQuantizedMolecule) : the molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        ansatz (Ansatze) : one of the supported ansatze.
        optimizer (function handle): a function defining the classical optimizer and its behavior.
        initial_var_params (str or array-like) : initial value for the classical optimizer.
        backend_options (dict) : parameters to build the tangelo.linq Simulator class.
        penalty_terms (dict): parameters for penalty terms to append to target qubit Hamiltonian (see penalty_terms
            for more details).
        ansatz_options (dict): parameters for the given ansatz (see given ansatz file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down.
            Default, False has alternating spin up/down ordering.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        verbose (bool): Flag for VQE verbosity.
        ref_states (list): The vector occupations of the reference configurations
        weights (array): The weights of the occupations
     """

    def __init__(self, opt_dict: dict):

        oo_options = {"tol": 1e-3,
                      "max_cycles": 15,
                      "n_oo_per_iter": 1}

        if "molecule" not in opt_dict:
            raise ValueError(f"A molecule must be provided for {self.__class__.__name__}")

        # remove SA-OO-VQE specific options before calling SA_VQESolver.__init__() and move values to oo_options
        opt_dict_sa_vqe = opt_dict.copy()
        for k, v in opt_dict.items():
            if k in oo_options:
                value = opt_dict_sa_vqe.pop(k)
                oo_options[k] = value

        # Initialization of SA_VQESOLVER will check if spurious dictionary items are present
        super().__init__(opt_dict_sa_vqe)

        # Add oo_options to attributes
        for k, v in oo_options.items():
            setattr(self, k, v)

        self.n_ref_states = len(self.ref_states)

        self.converged = False
        self.iteration = 0
        self.energies = list()
        # vqe_energies could include a penalty term contribution so will be different from energies calculated using rdms
        self.vqe_energies = list()

    def iterate(self):
        """Performs the SA-OO-VQE iterations.

        Each iteration, a SA-VQE minimization is performed followed by an orbital optimization. This process repeats until
        max_cycles are reached or the change in energy is less than tol.
        """
        for iter in range(self.max_cycles):
            vqe_energy = self.simulate()
            self.vqe_energies.append(vqe_energy)
            self.rdms = list()
            for reference_circuit in self.reference_circuits:
                self.rdms.append(self.get_rdm(self.optimal_var_params, ref_state=reference_circuit))
            energy_new = self.energy_from_rdms()
            if self.verbose:
                print(f"The State-Averaged VQE energy for iteration {iter} is: {energy_new}")
            if iter > 0 and abs(energy_new-self.energies[-1]) < self.tol:
                self.energies.append(energy_new)
                break
            for _ in range(self.n_oo_per_iter):
                u_mat = self.generate_oo_unitary()
                self.molecule.mean_field.mo_coeff = self.molecule.mean_field.mo_coeff @ u_mat
            self.energies.append(self.energy_from_rdms())
            if self.verbose:
                print(f"The State-Averaged Orbital Optimized energy for iteration {iter} is: {self.energies[-1]}")
            self.build()

    def energy_from_rdms(self):
        "Calculate energy from rdms generated from SA_VQESolver"
        fcore, foneint, ftwoint = self.molecule.get_full_space_integrals()
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
        for i in range(self.n_ref_states):
            one_rdm += self.rdms[i][0].real*self.weights[i]
            two_rdm += self.rdms[i][1].real*self.weights[i]/2

        for ti, t in enumerate(active_indices):
            for ui, u in enumerate(active_indices):
                active_energy += one_rdm[ti, ui]*(foneint[t, u] + v_mat[t, u])
                for vi, v in enumerate(active_indices):
                    for wi, w in enumerate(active_indices):
                        active_energy += two_rdm[ti, ui, vi, wi]*ftwoint[t, u, v, w]

        return fcore+core_constant+active_energy

    def generate_oo_unitary(self):
        """Generate the orbital optimization unitary that rotates the orbitals. It uses n_oo_per_iter Newton-Raphson steps
        with the Hessian calculated analytically.

        The unitary is generated using the method outlined in
        [1] Per E. M. Siegbahn, Jan Almlof, Anders Heiberg, and Bjorn O. Roos, "The complete active space SCF (CASSCF) method
        in a Newton-Raphson formulation with application to the HNO molecule", J. Chem. Phys. 74, 2384-2396 (1981)

        Returns:
            array: The unitary matrix that when applied to the mean-field coefficients reduces the state averaged energy
        """
        _, foneint, ftwoint = self.molecule.get_full_space_integrals()
        ftwoint = ftwoint.transpose(0, 3, 1, 2)
        occupied_indices = self.molecule.frozen_occupied
        unoccupied_indices = self.molecule.frozen_virtual
        active_indices = self.molecule.active_mos
        n_mos = self.molecule.n_mos
        n_active_mos = self.molecule.n_active_mos

        one_rdm = np.zeros((n_active_mos, n_active_mos))
        two_rdm = np.zeros((n_active_mos, n_active_mos, n_active_mos, n_active_mos))

        for i in range(self.n_ref_states):
            one_rdm += self.rdms[i][0].real*self.weights[i]
            two_rdm += self.rdms[i][1].real*self.weights[i]/2

        # The following calculation of the analytic Hessian and gradient are derived from [1]
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
        delta = np.eye(n_mos)
        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for j in occupied_indices:
                    for ui, u in enumerate(active_indices):
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[i, t, j, u] += 2*(two_rdm[ui, ti, vi, xi]*ftwoint[v, x, i, j]+(two_rdm[ui, xi, vi, ti] +
                                                         two_rdm[ui, xi, ti, vi])*ftwoint[v, i, x, j])
                            d2ed2x[i, t, j, u] += ((delta[t, v]-one_rdm[ti, vi])*(4*ftwoint[v, i, u, j]-ftwoint[u, i, v, j]-ftwoint[u, v, i, j]) +
                                                   (delta[u, v]-one_rdm[ui, vi])*(4*ftwoint[v, j, t, i]-ftwoint[t, j, v, i]-ftwoint[t, v, i, j]))
                        d2ed2x[i, t, j, u] += (one_rdm[ti, ui]*fi_mat[i, j]+delta[i, j]*(2*fi_mat[t, u]+2*fa_mat[t, u] -
                                               f_mat[t, u])-2*delta[t, u]*(fi_mat[i, j]+fa_mat[i, j]))

        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for j in occupied_indices:
                    for a in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            d2ed2x[i, t, j, a] += (2*delta[t, v]-one_rdm[ti, vi])*(4*ftwoint[a, j, v, i]-ftwoint[a, v, i, j]-ftwoint[a, i, v, j])
                        d2ed2x[i, t, j, a] += 2*delta[i, j]*(fi_mat[a, t]+fa_mat[a, t])-1/2*delta[i, j]*f_mat[t, a]

        for i in occupied_indices:
            for ti, t in enumerate(active_indices):
                for ui, u in enumerate(active_indices):
                    for a in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[i, t, u, a] += (-2)*(two_rdm[ti, ui, vi, xi]*ftwoint[a, i, v, x]+(two_rdm[ti, vi, ui, xi] +
                                                            two_rdm[ti, vi, xi, ui])*ftwoint[a, x, v, i])
                            d2ed2x[i, t, u, a] += one_rdm[ui, vi]*(4*ftwoint[a, v, t, i]-ftwoint[a, i, t, v]-ftwoint[a, t, v, i])
                        d2ed2x[i, t, u, a] += (-1)*one_rdm[ti, ui]*fi_mat[a, i]+delta[t, u]*(fi_mat[a, i]+fa_mat[a, i])

        for i in occupied_indices:
            for a in unoccupied_indices:
                for j in occupied_indices:
                    for b in unoccupied_indices:
                        d2ed2x[i, a, j, b] += (2*(4*ftwoint[a, i, b, j]-ftwoint[a, b, i, j]-ftwoint[a, j, b, i]) +
                                               2*delta[i, j]*(fi_mat[a, b]+fa_mat[a, b]) - 2*delta[a, b]*(fi_mat[i, j]+fa_mat[i, j]))

        for i in occupied_indices:
            for a in unoccupied_indices:
                for ti, t in enumerate(active_indices):
                    for b in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            d2ed2x[i, a, t, b] += one_rdm[ti, vi]*(4*ftwoint[a, i, b, v]-ftwoint[a, v, b, i]-ftwoint[a, b, v, i])
                        d2ed2x[i, a, t, b] += (-1)*delta[a, b]*(fi_mat[t, i]+fa_mat[t, i]) - 1/2*delta[a, b]*f_mat[t, i]

        for ti, t in enumerate(active_indices):
            for a in unoccupied_indices:
                for ui, u in enumerate(active_indices):
                    for b in unoccupied_indices:
                        for vi, v in enumerate(active_indices):
                            for xi, x in enumerate(active_indices):
                                d2ed2x[t, a, u, b] += 2*(two_rdm[ti, ui, vi, xi]*ftwoint[a, b, v, x]+(two_rdm[ti, xi, vi, ui] +
                                                         two_rdm[ti, xi, ui, vi])*ftwoint[a, x, b, v])
                                for yi, y in enumerate(active_indices):
                                    d2ed2x[t, a, u, b] -= (delta[a, b]*(two_rdm[ti, vi, xi, yi]*ftwoint[u, v, x, y] +
                                                           two_rdm[ui, vi, xi, yi]*ftwoint[t, v, x, y]))
                            d2ed2x[t, a, u, b] += (-1/2)*delta[a, b]*(one_rdm[ti, vi]*fi_mat[u, v]+one_rdm[ui, vi]*fi_mat[t, v])
                        d2ed2x[t, a, u, b] += one_rdm[ti, ui]*fi_mat[a, b]

        ivals = occupied_indices + active_indices
        jvals = active_indices + unoccupied_indices
        ij_list = list()
        for i in ivals:
            for j in jvals:
                if (j > i and not (i in active_indices and j in active_indices)):
                    ij_list.append([i, j])

        n_params = len(ij_list)
        hess = np.zeros((n_params, n_params))
        dedx = np.zeros(n_params)
        for p1, (i, j) in enumerate(ij_list):
            dedx[p1] = 2*(f_mat[i, j]-f_mat[j, i])
            for p2, (k, ll) in enumerate(ij_list):
                hess[p1, p2] = d2ed2x[i, j, k, ll]*2

        # Regularization to ensure all hessian eigenvalues are greater than zero
        E, _ = np.linalg.eigh(hess)
        fac = abs(E[0])*2 if E[0] < 0 else 0
        hess = hess + np.eye(n_params)*fac

        # Generate matrix elements for generating the unitary and calculate exponential of Skew-Hermitian matrix (a unitary)
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
