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

from itertools import product

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
                oo_options[k] = opt_dict_sa_vqe.pop(k)

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
                core_constant += (2 * ftwoint[i, i, j, j] - ftwoint[i, j, i, j])

        active_energy = 0
        v_mat = np.zeros((foneint.shape[0], foneint.shape[0]))
        for t, u in product(active_indices, repeat=2):
            for i in occupied_indices:
                v_mat[u, t] += 2 * ftwoint[i, i, t, u] - ftwoint[i, t, i, u]

        n_active_mos = self.molecule.n_active_mos
        one_rdm = np.zeros((n_active_mos, n_active_mos))
        two_rdm = np.zeros((n_active_mos, n_active_mos, n_active_mos, n_active_mos))
        for i in range(self.n_ref_states):
            one_rdm += self.rdms[i][0].real*self.weights[i]
            two_rdm += self.rdms[i][1].real*self.weights[i]/2

        for ti, t in enumerate(active_indices):
            for ui, u in enumerate(active_indices):
                active_energy += one_rdm[ti, ui] * (foneint[t, u] + v_mat[t, u])
                for vi, v in enumerate(active_indices):
                    for wi, w in enumerate(active_indices):
                        active_energy += two_rdm[ti, ui, vi, wi] * ftwoint[t, u, v, w]

        return fcore + core_constant + active_energy

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
        n_active_mos = self.molecule.n_active_mos
        n_mos = self.molecule.n_mos
        f = list(range(n_mos))
        oc = self.molecule.frozen_occupied
        ac = self.molecule.active_mos
        un = self.molecule.frozen_virtual
        n_oc = len(oc)
        n_ac = len(ac)
        n_un = len(un)

        one_rdm = np.zeros((n_active_mos, n_active_mos))
        two_rdm = np.zeros((n_active_mos, n_active_mos, n_active_mos, n_active_mos))

        for i in range(self.n_ref_states):
            one_rdm += self.rdms[i][0].real*self.weights[i]
            two_rdm += self.rdms[i][1].real*self.weights[i]/2

        # The following calculation of the analytic Hessian and gradient are derived from [1]
        f_mat = np.zeros((n_mos, n_mos))
        fi_mat = foneint.copy()
        fi_mat += 2*np.einsum("ijkk->ij", ftwoint[np.ix_(f, f, oc, oc)])-np.einsum("ikjk->ij", ftwoint[np.ix_(f, oc, f, oc)])
        fa_mat = np.einsum("tu,pqtu->pq", one_rdm, ftwoint[np.ix_(f, f, ac, ac)]) - 1/2*np.einsum("tu,ptqu->pq", one_rdm, ftwoint[np.ix_(f, ac, f, ac)])

        inds = np.ix_(oc, f)
        f_mat[inds] = 2*(fa_mat[inds]+fi_mat[inds])
        f_mat[np.ix_(ac, f)] += (np.einsum("tu,qu->tq", one_rdm, fi_mat[np.ix_(f, ac)])
                                 + 2*np.einsum("tuvx,quvx->tq", two_rdm, ftwoint[np.ix_(f, ac, ac, ac)]))

        d2ed2x = np.zeros((n_mos, n_mos, n_mos, n_mos))

        inds = np.ix_(oc, ac, oc, ac)
        indsoo = np.ix_(oc, oc)
        eye_m_one_rdm = np.eye(one_rdm.shape[0]) - one_rdm
        ftwointaoao = ftwoint[np.ix_(ac, oc, ac, oc)]
        ftwointaaoo = ftwoint[np.ix_(ac, ac, oc, oc)]
        d2ed2x[inds] += (2*(np.einsum("utvx,vxij->itju", two_rdm, ftwointaaoo)
                         + np.einsum("uxvt,vixj->itju", two_rdm + two_rdm.transpose([0, 1, 3, 2]), ftwointaoao))
                         + np.einsum("tv,viuj->itju", eye_m_one_rdm, 4*ftwointaoao-ftwointaoao.transpose([2, 1, 0, 3]) -
                                     ftwointaaoo.transpose([1, 2, 0, 3]))
                         + np.einsum("uv,vjti->itju", eye_m_one_rdm, 4*ftwointaoao-ftwointaoao.transpose([2, 1, 0, 3]) -
                                     ftwointaaoo.transpose([1, 3, 0, 2]))
                         + np.einsum("tu,ij->itju", one_rdm, fi_mat[indsoo]))
        indsaa = np.ix_(ac, ac)
        for i in oc:
            d2ed2x[np.ix_([i], ac, [i], ac)] += (2*fi_mat[indsaa]+2*fa_mat[indsaa]-f_mat[indsaa]).reshape((1, n_ac, 1, n_ac))
        for t in ac:
            d2ed2x[np.ix_(oc, [t], oc, [t])] -= 2*(fi_mat[indsoo] + fa_mat[indsoo]).reshape((n_oc, 1, n_oc, 1))

        inds = np.ix_(oc, ac, oc, un)
        eye_m_one_rdm = 2*np.eye(one_rdm.shape[0]) - one_rdm
        d2ed2x[inds] = np.einsum("tv,ajvi->itja", eye_m_one_rdm, 4*ftwoint[np.ix_(un, oc, ac, oc)] - ftwoint[np.ix_(un, ac, oc, oc)].transpose([0, 3, 1, 2])
                                 - ftwoint[np.ix_(un, oc, ac, oc)].transpose([0, 3, 2, 1]))
        indsua = np.ix_(un, ac)
        for i in oc:
            d2ed2x[np.ix_([i], ac, [i], un)] += (2*(fi_mat[indsua] + fa_mat[indsua]).transpose() - 1/2*f_mat[np.ix_(ac, un)]).reshape([1, n_ac, 1, n_un])

        inds = np.ix_(oc, ac, ac, un)
        ftwointuoaa = ftwoint[np.ix_(un, oc, ac, ac)]
        ftwointuaao = ftwoint[np.ix_(un, ac, ac, oc)]
        d2ed2x[inds] += ((-2)*(np.einsum("tuvx,aivx->itua", two_rdm, ftwointuoaa)
                         + np.einsum("tvux,axvi->itua", two_rdm+two_rdm.transpose([0, 1, 3, 2]), ftwointuaao))
                         + np.einsum("uv,avti->itua", one_rdm, 4*ftwointuaao - ftwointuoaa.transpose([0, 3, 2, 1]) -
                                     ftwointuaao.transpose([0, 2, 1, 3]))
                         - np.einsum("tu,ai->itua", one_rdm, fi_mat[np.ix_(un, oc)]))
        for t in ac:
            d2ed2x[np.ix_(oc, [t], [t], un)] += (fi_mat[np.ix_(un, oc)] + fa_mat[np.ix_(un, oc)]).transpose().reshape([n_oc, 1, 1, n_un])

        d2ed2x[np.ix_(oc, un, oc, un)] = 2*(4*ftwoint[np.ix_(un, oc, un, oc)].transpose([1, 0, 3, 2])
                                            - ftwoint[np.ix_(un, un, oc, oc)].transpose([2, 0, 3, 1])
                                            - ftwoint[np.ix_(un, oc, un, oc)].transpose([3, 0, 1, 2]))
        for i in oc:
            d2ed2x[np.ix_([i], un, [i], un)] += 2*(fi_mat[np.ix_(un, un)] + fa_mat[np.ix_(un, un)]).reshape([1, n_un, 1, n_un])
        for a in un:
            d2ed2x[np.ix_(oc, [a], oc, [a])] -= 2*(fi_mat[indsoo] + fa_mat[indsoo]).reshape([n_oc, 1, n_oc, 1])

        inds = np.ix_(oc, un, ac, un)
        d2ed2x[inds] = np.einsum("tv,aibv->iatb", one_rdm, 4*ftwoint[np.ix_(un, oc, un, ac)] -
                                 ftwoint[np.ix_(un, ac, un, oc)].transpose([0, 3, 2, 1]) -
                                 ftwoint[np.ix_(un, un, ac, oc)].transpose([0, 3, 1, 2]))
        for a in un:
            d2ed2x[np.ix_(oc, [a], ac, [a])] -= (fi_mat[np.ix_(ac, oc)] + fa_mat[np.ix_(ac, oc)] +
                                                 1/2*f_mat[np.ix_(ac, oc)]).transpose().reshape([n_oc, 1, n_ac, 1])

        inds = np.ix_(ac, un, ac, un)
        d2ed2x[inds] = (2*(np.einsum("tuvx,abvx->taub", two_rdm, ftwoint[np.ix_(un, un, ac, ac)]) +
                           np.einsum("txvu,axbv->taub", two_rdm+two_rdm.transpose([0, 1, 3, 2]), ftwoint[np.ix_(un, ac, un, ac)])) +
                        np.einsum("tu,ab->taub", one_rdm, fi_mat[np.ix_(un, un)]))
        ftwointaaaa = ftwoint[np.ix_(ac, ac, ac, ac)]
        fi_mataa = fi_mat[np.ix_(ac, ac)]
        for a in un:
            d2ed2x[np.ix_(ac, [a], ac, [a])] -= (np.einsum("tvxy,uvxy->tu", two_rdm, ftwointaaaa) +
                                                 np.einsum("uvxy,tvxy->tu", two_rdm, ftwointaaaa) +
                                                 1/2*(np.einsum("tv,uv->tu", one_rdm, fi_mataa) +
                                                      np.einsum("uv,tv->tu", one_rdm, fi_mataa))).reshape([n_ac, 1, n_ac, 1])

        ivals = oc + ac
        jvals = ac + un
        ij_list = list()
        for i in ivals:
            for j in jvals:
                if (j > i and not (i in ac and j in ac)):
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
        for p1, (i, j) in enumerate(ij_list):
            mat_rep[i, j] = knew[p1]
            mat_rep[j, i] = -knew[p1]

        return expm(-mat_rep)
