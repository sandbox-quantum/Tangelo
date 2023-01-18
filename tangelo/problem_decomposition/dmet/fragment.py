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

"""Module for data structure for DMET fragments."""

from dataclasses import dataclass, field
from itertools import product


import openfermion
from openfermion.chem.molecular_data import spinorb_from_spatial
import openfermion.ops.representations as reps
from openfermion.utils import down_index, up_index
import numpy as np
import pyscf
from pyscf import ao2mo

from tangelo.toolboxes.qubit_mappings.mapping_transform import get_fermion_operator


@dataclass
class SecondQuantizedDMETFragment:
    """Mimicking SecondQuantizedMolecule for DMET fragments. It has the minimal
    number of attributes and methods to be parsed by electronic solvers.
    """

    molecule: pyscf.gto
    mean_field: pyscf.scf

    # Fragment data computed by the DMET backend code. Useful when computing the
    # energy (converting embedded problem to full system energy).
    fock: np.array
    fock_frag_copy: np.array
    t_list: list
    one_ele: np.array
    two_ele: np.array

    uhf: bool

    n_active_electrons: int = field(init=False)
    n_active_sos: int = field(init=False)
    q: int = field(init=False)
    spin: int = field(init=False)

    basis: str = field(init=False)
    n_active_mos: int = field(init=False)
    frozen_mos: None = field(init=False)

    def __post_init__(self):
        self.n_active_electrons = self.molecule.nelectron
        self.n_active_ab_electrons = self.mean_field.nelec if self.uhf else self.n_active_electrons
        self.q = self.molecule.charge
        self.spin = self.molecule.spin
        self.active_spin = self.spin

        self.basis = self.molecule.basis
        self.n_active_mos = len(self.mean_field.mo_energy) if not self.uhf else (len(self.mean_field.mo_energy[0]), len(self.mean_field.mo_energy[1]))
        self.n_active_sos = 2*self.n_active_mos if not self.uhf else max(2*self.n_active_mos[0], 2*self.n_active_mos[1])

        self.frozen_mos = None

    @property
    def fermionic_hamiltonian(self):
        if self.uhf:
            return self._fermionic_hamiltonian_unrestricted()
        return self._fermionic_hamiltonian_restricted()

    def _fermionic_hamiltonian_restricted(self):
        """Computes the restricted Fermionic Hamiltonian, using the fragment
        attributes.

        Returns:
            FermionOperator: Self-explanatory.
        """
        mo_coeff = self.mean_field.mo_coeff

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(self.mean_field.mol.energy_nuc())

        # get_hcore is equivalent to int1e_kin + int1e_nuc.
        one_electron_integrals = mo_coeff.T @ self.mean_field.get_hcore() @ mo_coeff

        # Getting 2-body integrals in atomic and converting to molecular basis.
        two_electron_integrals = ao2mo.kernel(self.mean_field._eri, mo_coeff)
        two_electron_integrals = ao2mo.restore(1, two_electron_integrals, len(mo_coeff))

        # PQRS convention in openfermion:
        # h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # The convention is not the same with PySCF integrals. So, a change is
        # made before performing the truncation for frozen orbitals.
        two_electron_integrals = two_electron_integrals.transpose(0, 2, 3, 1)

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_electron_integrals, two_electron_integrals)
        fragment_hamiltonian = reps.InteractionOperator(core_constant, one_body_coefficients, 0.5 * two_body_coefficients)

        return get_fermion_operator(fragment_hamiltonian)

    def _fermionic_hamiltonian_unrestricted(self):
        """Computes the unrestricted Fermionic Hamiltonian, using the fragment
        attributes.

        Returns:
            FermionOperator: Self-explanatory.
        """
        mo_coeff = self.mean_field.mo_coeff

        # Molecular and atomic orbitals
        nao, nmo = mo_coeff[0].shape

        # Obtain Hcore Hamiltonian in atomic orbitals basis
        hcore = self.mean_field.get_hcore()

        # Obtain two-electron integral in atomic basis
        eri = ao2mo.restore(8, self.mean_field._eri, nao)

        # Create the placeholder for the matrices  one-electron matrix (alpha,
        # beta)
        hpq = []

        # Do the MO transformation step the mo coeff alpha and beta
        mo_a, mo_b = mo_coeff

        # MO transform the hcore
        hpq.append(mo_a.T.dot(hcore).dot(mo_a))
        hpq.append(mo_b.T.dot(hcore).dot(mo_b))

        # MO transform the two-electron integrals
        eri_a = ao2mo.incore.full(eri, mo_a)
        eri_b = ao2mo.incore.full(eri, mo_b)
        eri_ba = ao2mo.incore.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False)

        # Change the format of integrals (full)
        eri_a = ao2mo.restore(1, eri_a, nmo)
        eri_b = ao2mo.restore(1, eri_b, nmo)
        eri_ba = eri_ba.reshape(nmo, nmo, nmo, nmo)

        # Convert this into the order OpenFemion like to receive
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Corresponding to nuclear repulsion energy and static coulomb energy.
        core_constant = float(self.mean_field.mol.energy_nuc())

        one_body_integrals = hpq
        two_body_integrals = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        # Lets find the dimensions
        n_orb_a = one_body_integrals[0].shape[0]
        n_orb_b = one_body_integrals[1].shape[0]

        n_qubits = 2*max(n_orb_a, n_orb_b)

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits,) * 2)
        two_body_coefficients = np.zeros((n_qubits,) * 4)

        # aa
        for p, q in product(range(n_orb_a), repeat=2):
            pi, qi = up_index(p), up_index(q)
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[pi, qi] = one_body_integrals[0][p, q]
            for r, s in product(range(n_orb_a), repeat=2):
                two_body_coefficients[pi, qi, up_index(r), up_index(s)] = (two_body_integrals[0][p, q, r, s] / 2.)

        # bb
        for p, q in product(range(n_orb_b), repeat=2):
            pi, qi = down_index(p), down_index(q)
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[pi, qi] = one_body_integrals[1][p, q]
            for r, s in product(range(n_orb_b), repeat=2):
                two_body_coefficients[pi, qi, down_index(r), down_index(s)] = (two_body_integrals[2][p, q, r, s] / 2.)

        # abba
        for p, q, r, s in product(range(n_orb_a), range(n_orb_b), range(n_orb_b), range(n_orb_a)):
            two_body_coefficients[up_index(p), down_index(q), down_index(r), up_index(s)] = (two_body_integrals[1][p, q, r, s] / 2.)

        # baab
        for p, q, r, s in product(range(n_orb_b), range(n_orb_a), range(n_orb_a), range(n_orb_b)):
            two_body_coefficients[down_index(p), up_index(q), up_index(r), down_index(s)] = (two_body_integrals[1][q, p, s, r] / 2.)

        # Cast to InteractionOperator class and return.
        fragment_hamiltonian = openfermion.InteractionOperator(core_constant, one_body_coefficients, two_body_coefficients)

        return get_fermion_operator(fragment_hamiltonian)

    def to_pyscf(self, basis=None):
        """Method to output the PySCF molecule.

        Returns
            pyscf.gto.Mole: PySCF molecule.
        """
        return self.molecule
