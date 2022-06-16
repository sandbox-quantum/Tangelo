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

"""Module to regroup the list of possible atoms and built-in chemical groups for
broken link capping. The geometries for the capping groups are taken from the
NIST Chemistry WebBook database (https://webbook.nist.gov/chemistry/).
"""


elements = [
    "X",
    "H", "He", "Li", "Be",  "B",  "C",  "N",  "O",  "F", "Ne",
    "Na", "Mg", "Al", "Si",  "P",  "S", "Cl", "Ar",  "K", "Ca",
    "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",  "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te",  "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

chemical_groups = {
    "CH3": [  # From ethane.
        ["X", [  2.15410,  0.87450,  1.16590]],
        ["C", [  0.71660,  0.89800,  0.64250]],
        ["H", [  0.53970,  1.76660, -0.00250]],
        ["H", [  0.48990,  0.00050,  0.05510]],
        ["H", [ -0.00780,  0.94520,  1.46400]]
    ],
    "CF3": [  # From hexafluoroethane.
        ["X", [  2.03950,  1.01440,  0.00010]],
        ["C", [  0.49550,  1.01460, -0.00010]],
        ["F", [  0.04100, -0.24410,  0.00000]],
        ["F", [  0.04120,  1.64380, -1.09040]],
        ["F", [  0.04080,  1.64420,  1.08980]]
    ],
    "NH2": [  # From ammonia.
        ["X", [ -0.01300, -0.00270,  0.00580]],
        ["N", [  0.87140,  0.50430, -0.01000]],
        ["H", [  1.59990, -0.20890,  0.00580]],
        ["H", [  0.92660,  0.93550, -0.93240]]
    ]
}
