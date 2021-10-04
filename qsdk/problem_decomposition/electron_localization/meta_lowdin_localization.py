"""Perform Meta-Löwdin localization.

The orbital localization of the canonical orbitals  using Meta-Löwdin
localization is done here. `pyscf.lo` is used.

For details, refer to:
    - Q. Sun et al., JCTC 10, 3784-3790 (2014).
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
