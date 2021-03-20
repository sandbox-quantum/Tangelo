"""
    Set of functions and classes regarding classical computation of energies and electronic integrals before
    setting up electronic structure solvers.
"""

from enum import Enum
import warnings

from pyscf import scf
import openfermionpyscf


class MFType(Enum):
    """ Enum class to speficy the type of mean field / reference used to compute electronic integrals """
    RHF = 0,
    UHF = 1,
    ROHF = 2


def run_pyscf(molecule, run_scf=True, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True):
    """ Computes classical energies and electronic integrals using pyscf. To be replaced by more generic functions
     later on, with proper names """
    return openfermionpyscf.run_pyscf(molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci)


def prepare_mf_RHF(molecule):
    """ Prepares the RHF mean field for the input pyscf molecule, using pyscf for the calculations.
     Encapsulation. """

    mean_field = scf.RHF(molecule)
    mean_field.verbose = 0
    mean_field.scf()

    if not mean_field.converged:
        orb_temp = mean_field.mo_coeff
        occ_temp = mean_field.mo_occ
        nr = scf.newton(mean_field)
        mean_field = nr

    # Check the convergence of the mean field
    if not mean_field.converged:
        warnings.warn("OpenFermionParametricSolver simulating with mean field not converged.", RuntimeWarning)

    return mean_field
