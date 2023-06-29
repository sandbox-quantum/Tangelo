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

"""Perform NAO localization.

The orbital localization of the canonical orbitals using Natural Atomic Orbital
localization is done here. `pyscf.lo` is used.

For details, refer to:
    - Alan E. Reed, Robert B. Weinstock, and Frank Weinhold.
      Natural population analysis. J. Chem. Phys., 83(2):735-746, 1985.
"""


def nao_localization(mol, mf):
    """Localize the orbitals using NAO localization.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        mf (pyscf.scf): The mean field of the molecule.

    Returns:
        numpy.array: The localized orbitals (float64).
    """
    from pyscf.lo import orth
    return orth.orth_ao(mf, "NAO")
