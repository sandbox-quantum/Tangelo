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

"""Module to create some molecules used in the package unittest."""


from tangelo import Molecule, SecondQuantizedMolecule


# Dihydrogen.
xyz_H2 = [
    ("H", (0., 0., 0.)),
    ("H", (0., 0., 0.7414))
]
mol_H2_sto3g = SecondQuantizedMolecule(xyz_H2, q=0, spin=0, basis="sto-3g")
mol_H2_321g = SecondQuantizedMolecule(xyz_H2, q=0, spin=0, basis="3-21g")


# Tetrahydrogen.
xyz_H4 = [
    ("H", [0.7071067811865476, 0.0, 0.0]),
    ("H", [0.0, 0.7071067811865476, 0.0]),
    ("H", [-1.0071067811865476, 0.0, 0.0]),
    ("H", [0.0, -1.0071067811865476, 0.0])
]
mol_H4_sto3g = SecondQuantizedMolecule(xyz_H4, q=0, spin=0, basis="sto-3g")
mol_H4_sto3g_symm = SecondQuantizedMolecule(xyz_H4, q=0, spin=0, basis="sto-3g", symmetry=True)
mol_H4_minao = SecondQuantizedMolecule(xyz_H4, q=0, spin=0, basis="minao")
mol_H4_cation_sto3g = SecondQuantizedMolecule(xyz_H4, q=1, spin=1, basis="sto-3g")
mol_H4_doublecation_minao = SecondQuantizedMolecule(xyz_H4, q=2, spin=0, basis="minao")
mol_H4_doublecation_321g = SecondQuantizedMolecule(xyz_H4, q=2, spin=0, basis="3-21g")


# Decahydrogen.
xyz_H10 = [
    ("H", ( 0.970820393250,  0.000000000000, 0.)),
    ("H", ( 0.785410196625,  0.570633909777, 0.)),
    ("H", ( 0.300000000000,  0.923305061153, 0.)),
    ("H", (-0.300000000000,  0.923305061153, 0.)),
    ("H", (-0.785410196625,  0.570633909777, 0.)),
    ("H", (-0.970820393250,  0.000000000000, 0.)),
    ("H", (-0.785410196625, -0.570633909777, 0.)),
    ("H", (-0.300000000000, -0.923305061153, 0.)),
    ("H", ( 0.300000000000, -0.923305061153, 0.)),
    ("H", ( 0.785410196625, -0.570633909777, 0.))
]
mol_H10_minao = SecondQuantizedMolecule(xyz_H10, q=0, spin=0, basis="minao")
mol_H10_321g = SecondQuantizedMolecule(xyz_H10, q=0, spin=0, basis="3-21g")


# Water.
xyz_H2O = [
    ("O", (0., 0., 0.11779)),
    ("H", (0., 0.75545, -0.47116)),
    ("H", (0., -0.75545, -0.47116))
]
mol_H2O_sto3g = SecondQuantizedMolecule(xyz_H2O, q=0, spin=0, basis="sto-3g", frozen_orbitals=None)
mol_H2O_321g = SecondQuantizedMolecule(xyz_H2O, q=0, spin=0, basis="3-21g", frozen_orbitals=None)


# Sodium hydride.
xyz_NaH = [
    ("Na", (0., 0., 0.)),
    ("H", (0., 0., 1.91439))
]
mol_NaH_sto3g = SecondQuantizedMolecule(xyz_NaH, q=0, spin=0, basis="sto-3g", frozen_orbitals=None)


# Beryllium atom.
xyz_Be = [
    ("Be", (0., 0., 0.))
]
mol_Be_321g = SecondQuantizedMolecule(xyz_Be, q=0, spin=0, basis="3-21g", frozen_orbitals=None)


# Pyridine.
xyz_pyridine = [
    ("C", ( 1.3603,  0.0256, 0.)),
    ("C", ( 0.6971, -1.2020, 0.)),
    ("C", (-0.6944, -1.2184, 0.)),
    ("C", (-1.3895, -0.0129, 0.)),
    ("C", (-0.6712,  1.1834, 0.)),
    ("N", ( 0.6816,  1.1960, 0.)),
    ("H", ( 2.4530,  0.1083, 0.)),
    ("H", ( 1.2665, -2.1365, 0.)),
    ("H", (-1.2365, -2.1696, 0.)),
    ("H", (-2.4837,  0.0011, 0.)),
    ("H", (-1.1569,  2.1657, 0.))
]
mol_pyridine = Molecule(xyz_pyridine, q=0, spin=0)


# Phenylalanine amino acid.
xyz_PHE = [
    ("N", ( 0.7060, -1.9967, -0.0757)),
    ("C", ( 1.1211, -0.6335, -0.4814)),
    ("C", ( 0.6291,  0.4897,  0.4485)),
    ("C", (-0.8603,  0.6071,  0.4224)),
    ("C", (-1.4999,  1.1390, -0.6995)),
    ("C", (-2.8840,  1.2600, -0.7219)),
    ("C", (-3.6384,  0.8545,  0.3747)),
    ("C", (-3.0052,  0.3278,  1.4949)),
    ("C", (-1.6202,  0.2033,  1.5209)),
    ("C", ( 2.6429, -0.5911, -0.5338)),
    ("O", ( 3.1604, -0.2029, -1.7213)),
    ("O", ( 3.4477, -0.8409,  0.3447)),
    ("H", (-0.2916, -2.0354, -0.0544)),
    ("H", ( 1.0653, -2.2124,  0.8310)),
    ("H", ( 0.6990, -0.4698, -1.5067)),
    ("H", ( 1.0737,  1.4535,  0.1289)),
    ("H", ( 0.9896,  0.3214,  1.4846)),
    ("H", (-0.9058,  1.4624, -1.5623)),
    ("H", (-3.3807,  1.6765, -1.6044)),
    ("H", (-4.7288,  0.9516,  0.3559)),
    ("H", (-3.5968,  0.0108,  2.3601)),
    ("H", (-1.1260, -0.2065,  2.4095)),
    ("H", ( 4.1118, -0.2131, -1.6830))
]


# Ethane (H3CCH3).
xyz_ethane = [
    ("C", ( 0.7166,  0.8980,  0.6425)),
    ("H", ( 0.5397,  1.7666, -0.0025)),
    ("H", ( 0.4899,  0.0005,  0.0551)),
    ("H", (-0.0078,  0.9452,  1.4640)),
    ("C", ( 2.1541,  0.8745,  1.1659)),
    ("H", ( 2.3313,  0.0053,  1.8100)),
    ("H", ( 2.8785,  0.8284,  0.3444)),
    ("H", ( 2.3805,  1.7715,  1.7542))
]
