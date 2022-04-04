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

"""Module to generate the rotations to convert the 2-body integrals into a sum of diagonal terms

Refs:
    [1] Mario Motta, Erika Ye, Jarrod R. McClean, Zhendong Li, Austin J. Minnich, Ryan Babbush, Garnet Kin-Lic Chan
    "Low rank representations for quantum simulation of electronic structure". https://arxiv.org/abs/1808.02625"""

from openfermion import givens_decomposition_square
from openfermion.circuits.low_rank import low_rank_two_body_decomposition, prepare_one_body_squared_evolution
from openfermion.chem.molecular_data import spinorb_from_spatial
import numpy as np

from tangelo.linq import Gate
from tangelo import SecondQuantizedMolecule
from tangelo.toolboxes.operators import FermionOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping


def bogoliubov_transform(umat, add_phase=True):
    """Generate the gates that implement the orbital rotation defined by a given unitary.

    The given matrix is decomposed into umat=DU where U is unitary D is a diagonal matrix.
    If umat is unitary, the values of D will be exp(1j*theta) where theta is real
    and can be incorporated using add_phase=True.
    If umat is non-unitary, the values of D will be m*exp(1j*theta) where m and theta are real.
    m is ignored with only the phase (theta) incorporated in the circuit if add_phase=True.

    This implementation uses the circuit outlined in https://arxiv.org/abs/1711.05395

    Args:
        umat (array): The square unitary matrix that descibes the basis rotation
        add_phase (bool): If True, adds PHASE gates according to givens decomposition values D

    Returns:
        list of Gate: The sequence of Gates that implement the orbital rotation
    """
    gs = givens_decomposition_square(umat)
    gates = []
    if add_phase:
        for i, phase in enumerate(gs[1]):
            gates += [Gate('PHASE', i, parameter=-np.arctan2(np.imag(phase), np.real(phase)))]
    for ele in reversed(gs[0]):
        for el1 in reversed(ele):
            i, j = el1[0:2]
            gates += [Gate('CNOT', i, j), Gate('CRY', j, i, parameter=2*el1[2]), Gate('CNOT', i, j), Gate('PHASE', j, parameter=-el1[3])]

    return gates[::-1]


class OrbitalRotations():
    """The class that holds the various rotations, operators and coefficients

    Attributes:
        rotation_gates (list): The gates to apply to rotate the basis to a diagonal form for a given operator
        qubit_operators (list): The qubit operators to measure from qubit_mapping="JW" and up_then_down=False. Each is diagonal in the Z basis.
        one_body_coefficients (list): The one-body coefficients corresponding to each operator
        constants (list): The constants corresponding to each operator
        two_body_coefficients (list): The two-body coefficients corresponding to each operator
    """

    def __init__(self):
        self.rotation_gates = list()
        self.qubit_operators = list()
        self.one_body_coefficients = list()
        self.constants = list()
        self.two_body_coefficients = list()

    def add_elements(self, rot_mat, constant=0., one_body_coefs=None, two_body_coefs=None):
        """Add elements to the class for each term to diagonalize a portion of the Hamiltonian

        Args:
            rot_mat (array): The unitary rotation matrix
            constant (float): A constant value for this element
            one_body_coefs (array): The diagonal coefficients
            two_body_coefs (array): The two-body coefficients
        """
        self.rotation_gates.append(bogoliubov_transform(rot_mat))
        fe_op = FermionOperator((), coefficient=constant)
        if one_body_coefs is not None:
            for p, value in enumerate(one_body_coefs):
                fe_op += FermionOperator(((p, 1), (p, 0)), value)

        if two_body_coefs is not None:
            num_vals = np.array(two_body_coefs).shape[0]
            for p in range(num_vals):
                fe_op += FermionOperator(((p, 1), (p, 0), (p, 1), (p, 0)), two_body_coefs[p, p])
                for q in range(p+1, num_vals):
                    fe_op += FermionOperator(((p, 1), (p, 0), (q, 1), (q, 0)), two_body_coefs[p, q]/2)
                    fe_op += FermionOperator(((q, 1), (q, 0), (p, 1), (p, 0)), two_body_coefs[q, p]/2)
        self.qubit_operators.append(fermion_to_qubit_mapping(fe_op, "JW", up_then_down=False))
        self.one_body_coefficients.append(one_body_coefs)
        self.two_body_coefficients.append(two_body_coefs)
        self.constants.append(constant)

    @property
    def n_rotations(self):
        return len(self.constants)


def get_orbital_rotations(molecule: SecondQuantizedMolecule):
    """Generate the gates that rotate the orbitals such that each operator is diagonal.

    The orbital rotation gates generated are only applicable for qubit_mapping="JW" and up_then_down=False.

    Args:
        molecule (SecondQuantizedMolecule): The molecule for which the diagonal representation is requested

    Returns:
        OrbitalRotations: The class that contains the list of transformation gates and operators to measure
    """

    core_constant, one_body, two_body = molecule.get_active_space_integrals()

    eigenvalues, one_body_squares, one_body_correction, _ = low_rank_two_body_decomposition(two_body, truncation_threshold=1.e-20, spin_basis=False)

    scaled_density_density_matrices = list()
    basis_change_matrices = list()
    for j, eigenvalue in enumerate(eigenvalues):
        one_body_squares[j] = (one_body_squares[j].T.conj() + one_body_squares[j])/2
        dd_mat, basis_change_mat = prepare_one_body_squared_evolution(one_body_squares[j])
        scaled_density_density_matrices.append(np.real(eigenvalue*dd_mat))
        basis_change_matrices.append(basis_change_mat)

    # Generate basis rotation matrix and eigenvalues for one-body term
    one_body_t, _ = spinorb_from_spatial(one_body, two_body)
    one_body_coefs = one_body_t + one_body_correction
    eigs, umat = np.linalg.eigh(one_body_coefs)

    orb_rots = OrbitalRotations()
    orb_rots.add_elements(umat.T.conj(), constant=core_constant, one_body_coefs=eigs)

    # Generate elements for two-body terms
    for j in range(len(eigenvalues)):
        basis_change_mat = basis_change_matrices[j]
        two_body_coefs = scaled_density_density_matrices[j]
        orb_rots.add_elements(basis_change_mat, two_body_coefs=two_body_coefs)

    return orb_rots
