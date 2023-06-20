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

"""Class performing electronic structure calculation employing the CCSD method.
"""

from typing import Union, Type

import numpy as np
from sympy.combinatorics.permutations import Permutation

from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.toolboxes.molecular_computation import IntegralSolverPsi4, IntegralSolverPySCF
from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.helpers.utils import installed_chem_backends, is_package_installed

if 'pyscf' in installed_chem_backends:
    default_ccsd_solver = 'pyscf'
elif 'psi4' in installed_chem_backends:
    default_ccsd_solver = 'psi4'
else:
    default_ccsd_solver = None


class CCSDSolverPySCF(ElectronicStructureSolver):
    """Uses the CCSD method to solve the electronic structure problem, through
    pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        cc_fragment (pyscf.cc.CCSD): The coupled-cluster object.
        mean_field (pyscf.scf.RHF): The mean field of the molecule.
        frozen (list or int): Frozen molecular orbitals.
    """

    def __init__(self, molecule):
        if not is_package_installed("pyscf"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")
        from pyscf import cc

        self.cc = cc
        self.cc_fragment = None

        self.spin = molecule.spin

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos
        self.uhf = molecule.uhf

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: CCSD energy.
        """
        # Execute CCSD calculation
        self.cc_fragment = self.cc.CCSD(self.mean_field, frozen=self.frozen)
        self.cc_fragment.verbose = 0
        self.cc_fragment.conv_tol = 1e-9
        self.cc_fragment.conv_tol_normt = 1e-7

        correlation_energy, _, _ = self.cc_fragment.ccsd()
        total_energy = self.mean_field.e_tot + correlation_energy

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices. The CCSD
        lambda equation will be solved for calculating the RDMs.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If no simulation has been run.
        """
        from pyscf import lib
        from pyscf.cc.ccsd_rdm import _make_rdm1, _make_rdm2, _gamma1_intermediates, _gamma2_outcore
        from pyscf.cc.uccsd_rdm import (_make_rdm1 as _umake_rdm1, _make_rdm2 as _umake_rdm2,
                                        _gamma1_intermediates as _ugamma1_intermediates, _gamma2_outcore as _ugamma2_outcore)

        # Check if CCSD calculation is performed
        if self.cc_fragment is None:
            raise RuntimeError("CCSDSolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        t1 = self.cc_fragment.t1
        t2 = self.cc_fragment.t2
        l1, l2 = self.cc_fragment.solve_lambda(t1, t2)

        if self.spin == 0 and not self.uhf:
            d1 = _gamma1_intermediates(self.cc_fragment, t1, t2, l1, l2)
            f = lib.H5TmpFile()
            d2 = _gamma2_outcore(self.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = _make_rdm1(self.cc_fragment, d1, with_frozen=False)
            two_rdm = _make_rdm2(self.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)
        else:
            d1 = _ugamma1_intermediates(self.cc_fragment, t1, t2, l1, l2)
            f = lib.H5TmpFile()
            d2 = _ugamma2_outcore(self.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = _umake_rdm1(self.cc_fragment, d1, with_frozen=False)
            two_rdm = _umake_rdm2(self.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)

            if not self.uhf:
                one_rdm = np.sum(one_rdm, axis=0)
                two_rdm = np.sum((two_rdm[0], 2*two_rdm[1], two_rdm[2]), axis=0)

        return one_rdm, two_rdm


class CCSDSolverPsi4(ElectronicStructureSolver):
    """ Uses the CCSD method to solve the electronic structure problem,
    through Psi4.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        ccwfn (psi4.core.CCWavefunction): The CCSD wavefunction (float64).
        backend (psi4): The psi4 module
        molecule (SecondQuantizedMolecule): The molecule with symmetry=False
    """

    def __init__(self, molecule: SecondQuantizedMolecule):
        if not is_package_installed("psi4"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the Psi4 package")

        import psi4
        self.backend = psi4
        self.backend.core.clean_options()
        self.backend.core.clean()
        self.backend.core.clean_variables()
        self.ccwfn = None

        self.n_frozen_vir = len(molecule.frozen_virtual) if not molecule.uhf else len(molecule.frozen_virtual[0])
        self.n_frozen_occ = len(molecule.frozen_occupied) if not molecule.uhf else len(molecule.frozen_occupied[0])
        if not molecule.uhf:
            self.ref = 'rhf' if molecule.spin == 0 else 'rohf'
        else:
            self.ref = 'uhf'
            self.n_frozen_vir_b = len(molecule.frozen_virtual[1])
            self.n_frozen_occ_b = len(molecule.frozen_occupied[1])
            if (self.n_frozen_vir, self.n_frozen_occ) != (self.n_frozen_vir_b, self.n_frozen_occ_b):
                raise ValueError(f"Tangelo does not support unequal number of alpha v. beta frozen or virtual orbitals"
                                 f"with a UHF reference in {self.__class__.__name__}")

        # Frozen orbitals must be declared before calling compute_mean_field to be saved in ref_wfn for Psi4 ccsd.
        intsolve = IntegralSolverPsi4()
        self.backend.set_options({'basis': molecule.basis, 'frozen_docc': [self.n_frozen_occ], 'frozen_uocc': [self.n_frozen_vir],
                                  'reference': self.ref})
        self.molecule = SecondQuantizedMolecule(xyz=molecule.xyz, q=molecule.q, spin=molecule.spin,
                                                solver=intsolve,
                                                basis=molecule.basis,
                                                ecp=molecule.ecp,
                                                symmetry=False,
                                                uhf=molecule.uhf,
                                                frozen_orbitals=molecule.frozen_orbitals)
        self.basis = molecule.basis

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total CCSD energy.
        """
        # Copy reference wavefunction and swap orbitals to obtain correct active space if necessary
        wfn = self.backend.core.Wavefunction(self.molecule.solver.mol_nosym, self.molecule.solver.wfn.basisset())
        wfn.deep_copy(self.molecule.solver.wfn)
        if self.n_frozen_occ or self.n_frozen_vir:
            if not self.molecule.uhf:
                mo_order = self.molecule.frozen_occupied + self.molecule.active_occupied + self.molecule.active_virtual + self.molecule.frozen_virtual
                # Obtain swap operations that will take the unordered list back to ordered with the correct active space in the middle.
                swap_ops = Permutation(mo_order).transpositions()
                for swap_op in swap_ops:
                    wfn.Ca().rotate_columns(0, swap_op[0], swap_op[1], np.deg2rad(90))

            else:

                # Obtain swap operations that will take the unordered list back to ordered with the correct active space in the middle.
                mo_order = (self.molecule.frozen_occupied[0] + self.molecule.active_occupied[0]
                            + self.molecule.active_virtual[0] + self.molecule.frozen_virtual[0])
                swap_ops = Permutation(mo_order).transpositions()
                for swap_op in swap_ops:
                    wfn.Ca().rotate_columns(0, swap_op[0], swap_op[1], np.deg2rad(90))

                # Repeat for Beta orbitals
                mo_order_b = (self.molecule.frozen_occupied[1] + self.molecule.active_occupied[1]
                              + self.molecule.active_virtual[1] + self.molecule.frozen_virtual[1])
                swap_ops = Permutation(mo_order_b).transpositions()
                for swap_op in swap_ops:
                    wfn.Cb().rotate_columns(0, swap_op[0], swap_op[1], np.deg2rad(90))

        self.backend.set_options({'basis': self.basis, 'frozen_docc': [self.n_frozen_occ], 'frozen_uocc': [self.n_frozen_vir],
                                  'qc_module': 'ccenergy', 'reference': self.ref})
        energy, self.ccwfn = self.backend.energy('ccsd', molecule=self.molecule.solver.mol,
                                                 basis=self.basis, return_wfn=True, ref_wfn=wfn)
        return energy

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returning RDMS from a CCSD calculation in Psi4 is not implemented at this time.

        It may be possible to obtain the one-rdm by running a psi4 CCSD gradient calculation
        (https://forum.psicode.org/t/saving-ccsd-density-for-read-in/2416/2)
        Another option to obtain the one-rdm is to use pycc (https://github.com/CrawfordGroup/pycc)

        Raises:
            NotImplementedError: Returning RDMs from Psi4 in Tangelo is not supported at this time.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not currently support returning RDMs")


ccsd_solver_dict = {'pyscf': CCSDSolverPySCF, 'psi4': CCSDSolverPsi4}


def get_ccsd_solver(molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_ccsd_solver, **solver_kwargs):
    """Return requested target CCSDSolverName object.

    Args:
        molecule (SecondQuantizedMolecule) : Molecule
        solver (string or Type[ElectronicStructureSolver] or None): Supported string identifiers can be found in
            ccsd_solver_dict (from tangelo.algorithms.classical.ccsd_solver). Can also provide a user-defined backend (child to ElectronicStructureSolver class)
        solver_kwargs: Other arguments that could be passed to a target. Examples are solver type (e.g. lambdacc, fnocc), Convergence options etc.
     """

    if solver is None:
        if isinstance(molecule.solver, IntegralSolverPySCF):
            solver = CCSDSolverPySCF
        elif isinstance(molecule.solver, IntegralSolverPsi4):
            solver = CCSDSolverPsi4
        elif default_ccsd_solver is not None:
            solver = default_ccsd_solver
        else:
            raise ModuleNotFoundError(f"One of the backends for {list(ccsd_solver_dict.keys())} needs to be installed to use a CCSDSolver"
                                      "without providing a user-defined implementation.")

    # If target is a string use target_dict to return built-in backend
    elif isinstance(solver, str):
        try:
            solver = ccsd_solver_dict[solver.lower()]
        except KeyError:
            raise ValueError(f"Error: backend {solver} not supported. Available built-in options: {list(ccsd_solver_dict.keys())}")
    elif not issubclass(solver, ElectronicStructureSolver):
        raise TypeError(f"Target must be a str or a subclass of ElectronicStructureSolver but received class {type(solver).__name__}")

    return solver(molecule, **solver_kwargs)


class CCSDSolver(ElectronicStructureSolver):
    """Uses the Full CI method to solve the electronic structure problem.

    Args:
        molecule (SecondQuantizedMolecule) : Molecule
        solver (string or Type[ElectronicStructureSolver] or None): Supported string identifiers can be found in
            available_ccsd_solvers (from tangelo.algorithms.classical.ccsd_solver). Can also provide a user-defined CCSD implementation
            (child to ElectronicStructureSolver class)
        solver_kwargs: Other arguments that could be passed to a target. Examples are solver type (e.g. lambdacc, fnocc), Convergence options etc.

    Attributes:
        solver (Type[ElectronicStructureSolver]): The backend specific CCSD solver
    """

    def __init__(self, molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_ccsd_solver, **solver_kwargs):
        self.solver = get_ccsd_solver(molecule, solver, **solver_kwargs)

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total CCSD energy.
        """
        return self.solver.simulate()

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If method "simulate" hasn't been run.
        """
        return self.solver.get_rdm()
