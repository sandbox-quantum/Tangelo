# Copyright SandboxAQ 2021-2024.
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

"""Construct the orbital list for the fragments.

Construct the orbital list, showing how much orbitals to be included for each
fragment, from the atom list, showing how many atoms included in each fragment.
"""


def dmet_fragment_constructor(mol, atom_list, n_fragment):
    """Construct orbital list.

    Make a list of number of orbitals for each fragment while obtaining the list
    if we consider combining fragments.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate (The full molecule).
        atom_list (list): The atom list for each fragment (int).
        n_fragment (list): Number of atoms per fragment (int).

    Returns:
        list: The number of orbitals for each fragment (int).
        list: List of lists of the minimum and maximum orbital label for each
            fragment (int).
        list: The new atom list for each fragment (int).
    """

    # Make a new atom list based on how many fragments for DMET calculation
    if n_fragment == 0:
        atom_list2 = atom_list
    else:
        # Calculate the number of DMET calculations
        n_new_fragment = int(len(atom_list)/(n_fragment+1))
        atom_list2 = []

        # Define the number of atoms per DMET calculation
        for i in range(n_new_fragment):
            num = 0
            for j in range(n_fragment + 1):
                k = (n_fragment + 1) * i + j
                num += atom_list[k]
            atom_list2.append(num)

    # Initialize the list of the number of orbitals
    orb_list = []
    orb_list2 = []
    isum = 0
    isum2 = -1
    iorb = 0
    jorb = 0

    # Calculate the number of orbitals for each atom
    for i in atom_list2:
        itemp = 0
        isum2 += i
        for total_basis in mol.spheric_labels():
            item = total_basis.split()
            item0 = int(item[0])
            if (isum <= item0 <= isum2):
                itemp += 1
        isum += i
        jorb += itemp
        orb_list.append(itemp)
        orb_list2.append([iorb, jorb])
        iorb += itemp

    return orb_list, orb_list2, atom_list2
