#   Copyright 2019 1QBit
#   
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Perform Meta-Löwdin localization.

The orbital localization of the canonical orbitals 
using Meta-Löwdin localization is done here.
`pyscf.lo` is used.

For details, refer to:
Q. Sun et al., JCTC 10, 3784-3790 (2014).

"""

from pyscf.lo import orth

def meta_lowdin_localization(mol, mf):
    """Localize the orbitals using Meta-Löwdin localization.

    Args:
        mol (pyscf.gto.Mole): The molecule to simulate.
        mf (pyscf.scf.RHF): The mean field of the molecule.

    Returns:
        numpy.array: The localized orbitals (float64).
    """
    return orth.orth_ao(mol, "meta_lowdin")

