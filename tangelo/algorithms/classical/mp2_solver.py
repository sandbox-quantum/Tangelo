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

"""Define electronic structure solver employing the Moller-Plesset perturbation theory
to second order (MP2) method.
"""

from typing import Union, Type
from itertools import combinations, product
from math import ceil

import numpy as np

from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.helpers.utils import installed_chem_backends, is_package_installed
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.toolboxes.molecular_computation import IntegralSolverPsi4, IntegralSolverPySCF
from tangelo.toolboxes.ansatz_generator._unitary_cc_openshell import uccsd_openshell_get_packed_amplitudes

if 'pyscf' in installed_chem_backends:
    default_mp2_solver = 'pyscf'
elif 'psi4' in installed_chem_backends:
    default_mp2_solver = 'psi4'
else:
    default_mp2_solver = None


class MP2SolverPySCF(ElectronicStructureSolver):
    """Uses the Second-order Moller-Plesset perturbation theory (MP2) method to solve the electronic structure problem,
    through pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        mp2_fragment (pyscf.mp.MP2): The coupled-cluster object.
        mean_field (pyscf.scf.RHF): The mean field of the molecule.
        frozen (list or int): Frozen molecular orbitals.
    """

    def __init__(self, molecule):
        if not is_package_installed("pyscf"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")
        from pyscf import mp

        self.mp = mp
        self.mp2_fragment = None

        self.spin = molecule.spin

        self.mean_field = molecule.mean_field
        self.frozen = molecule.frozen_mos
        self.uhf = molecule.uhf

        # Define variables used to transform the MP2 parameters into an ordered
        # list of parameters with single and double excitations.
        if self.spin != 0 or self.uhf:
            self.n_alpha, self.n_beta = molecule.n_active_ab_electrons
            self.n_active_moa, self.n_active_mob = molecule.n_active_mos if self.uhf else (molecule.n_active_mos,)*2
        else:
            self.n_occupied = ceil(molecule.n_active_electrons / 2)
            self.n_virtual = molecule.n_active_mos - self.n_occupied

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: MP2 energy.
        """

        # Execute MP2 calculation
        if self.uhf:
            self.mp2_fragment = self.mp.UMP2(self.mean_field, frozen=self.frozen)
        else:
            self.mp2_fragment = self.mp.RMP2(self.mean_field, frozen=self.frozen)

        self.mp2_fragment.verbose = 0
        _, self.mp2_t2 = self.mp2_fragment.kernel()

        total_energy = self.mp2_fragment.e_tot

        return total_energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If no simulation has been run.
        """

        # Check if MP2 has been performed
        if self.mp2_fragment is None:
            raise RuntimeError(f"{self.__class__.__name__}: Cannot retrieve RDM. Please run the 'simulate' method first")
        if self.frozen is not None:
            raise RuntimeError(f"{self.__class__.__name__}: RDM calculation is not implemented with frozen orbitals.")

        one_rdm = self.mp2_fragment.make_rdm1()
        two_rdm = self.mp2_fragment.make_rdm2()

        return one_rdm, two_rdm

    def get_mp2_amplitudes(self):
        """Compute the double amplitudes from the MP2 perturbative method, and
        then reorder the elements into a dense list. The single (T1) amplitudes
        are set to a small non-zero value. The ordering is single, double
        (diagonal), double (non-diagonal).

        Returns:
            list of float: The electronic excitation amplitudes.
        """

        # Check if MP2 has been performed.
        if self.mp2_fragment is None:
            raise RuntimeError(f"{self.__class__.__name__}: Cannot retrieve MP2 parameters. Please run the 'simulate' method first")

        if self.spin != 0 or self.uhf:
            # Reorder the T2 amplitudes in a dense list.
            mp2_params = uccsd_openshell_get_packed_amplitudes(
                self.mp2_t2[0],  # aa
                self.mp2_t2[2],  # bb
                self.mp2_t2[1],  # ab
                self.n_alpha,
                self.n_beta,
                self.n_active_moa,
                self.n_active_mob
            )
        else:
            # Get singles amplitude. Just get "up" amplitude, since "down" should be the same
            singles = [2.e-5] * (self.n_virtual * self.n_occupied)

            # Get singles and doubles amplitudes associated with one spatial occupied-virtual pair
            doubles_1 = [-self.mp2_t2[q, q, p, p]/2. if (abs(-self.mp2_t2[q, q, p, p]/2.) > 1e-15) else 0.
                         for p, q in product(range(self.n_virtual), range(self.n_occupied))]

            # Get doubles amplitudes associated with two spatial occupied-virtual pairs
            doubles_2 = [-self.mp2_t2[q, s, p, r] for (p, q), (r, s)
                         in combinations(product(range(self.n_virtual), range(self.n_occupied)), 2)]

            mp2_params = singles + doubles_1 + doubles_2

        return mp2_params


class MP2SolverPsi4(ElectronicStructureSolver):
    """ Uses the MP2 method to solve the electronic structure problem, through Psi4.

    Only supports frozen core (active) orbitals sequentially from bottom (top) of energy ordering.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        mp2wfn (psi4.core.Wavefunction): The Psi4 Wavefunction returned from an mp2 calculation.
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
        self.mp2wfn = None

        self.molecule = SecondQuantizedMolecule(xyz=molecule.xyz, q=molecule.q, spin=molecule.spin,
                                                solver=IntegralSolverPsi4(),
                                                basis=molecule.basis,
                                                ecp=molecule.ecp,
                                                symmetry=False,
                                                uhf=molecule.uhf,
                                                frozen_orbitals=molecule.frozen_orbitals)
        self.basis = molecule.basis

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total MP2 energy.
        """
        n_frozen_vir = len(self.molecule.frozen_virtual)
        n_frozen_occ = len(self.molecule.frozen_occupied)

        if n_frozen_occ or n_frozen_vir:
            if self.molecule.uhf:
                if (set(self.molecule.frozen_occupied[0]) != set(self.molecule.frozen_occupied[1]) or
                   set(self.molecule.frozen_virtual[0]) != set(self.molecule.frozen_virtual)):
                    raise ValueError(f"Only identical frozen orbitals for alpha and beta are supported in {self.__class__.__name__}")
            focc = np.array(self.molecule.frozen_occupied)
            fvir = np.array(self.molecule.frozen_virtual)
            if np.any(focc > n_frozen_occ-1) or np.any(fvir < self.molecule.n_mos-n_frozen_vir):
                raise ValueError(f"{self.__class__.__name__} does not support freezing interior orbitals")

        if not self.molecule.uhf:
            ref = 'rhf' if self.molecule.spin == 0 else 'rohf'
        else:
            ref = 'uhf'
        self.backend.set_options({'basis': self.basis, 'mcscf_maxiter': 300, 'mcscf_diis_start': 20,
                                  'opdm': True, 'tpdm': True, 'frozen_docc': [n_frozen_occ], 'frozen_uocc': [n_frozen_vir],
                                  'reference': ref})

        energy, self.mp2wfn = self.backend.energy('mp2', molecule=self.molecule.solver.mol,
                                                  basis=self.basis, return_wfn=True)
        return energy

    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices.

        Obtaining MP2 rdms from Psi4 is not currently supported in Tangelo.

        Using https://github.com/psi4/psi4numpy/blob/master/Tutorials/10_Orbital_Optimized_Methods/10a_orbital-optimized-mp2.ipynb
        should return appropriate RDMs for a closed shell RHF reference.

        Raises:
            NotImplementedError: Not implemented at this time"""
        raise NotImplementedError("Returning MP2 rdms from Psi4 is not currently supported in Tangelo")

    def get_mp2_amplitudes(self):
        """Compute the double amplitudes from the MP2 perturbative method, and
        then reorder the elements into a dense list. The single (T1) amplitudes
        are set to a small non-zero value. The ordering is single, double
        (diagonal), double (non-diagonal).

        Returning MP2 amplitudes from Psi4 is not currently supported in Tangelo

        Using https://github.com/psi4/psi4numpy/blob/master/Tutorials/10_Orbital_Optimized_Methods/10a_orbital-optimized-mp2.ipynb
        should return appropriate amplitudes for a closed shell RHF reference.

        Raises:
            NotImplementedError: Not implemented at this time"""
        raise NotImplementedError("Returning MP2 amplitudes from Psi4 is not currently supported in Tangelo")


available_mp2_solvers = {'pyscf': MP2SolverPySCF, 'psi4': MP2SolverPsi4}


def get_mp2_solver(molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_mp2_solver, **solver_kwargs):
    """Return requested target MP2SolverName object.

    Args:
        molecule (SecondQuantizedMolecule) : Molecule
        solver (string or Type[ElectronicStructureSolver] or None): Supported string identifiers can be found in
            available_mp2_solvers (see mp2_solver.py). Can also provide a user-defined MP2 implementation
            (child to ElectronicStructureSolver class)
        solver_kwargs: Other arguments that could be passed to a target. Examples are solver type (e.g. mcscf, mp2), Convergence options etc.

    Raises:
        ModuleNoyFoundError: No solver is specified and a user defined IntegralSolver was used in molecule.
        ValueError: The specified solver str is not one of the available_mp2_solvers (see mp2_solver.py)
        TypeError: The specified solver was not a string or sub class of ElectronicStructureSolver.
     """

    if solver is None:
        if isinstance(molecule.solver, IntegralSolverPySCF):
            solver = MP2SolverPySCF
        elif isinstance(molecule.solver, IntegralSolverPsi4):
            solver = MP2SolverPsi4
        elif default_mp2_solver is not None:
            solver = default_mp2_solver
        else:
            raise ModuleNotFoundError(f"One of the backends for {list(available_mp2_solvers.keys())} needs to be installed to use MP2Solver"
                                      "without providing a user-defined implementation.")

    # If target is a string use target_dict to return built-in backend
    elif isinstance(solver, str):
        try:
            solver = available_mp2_solvers[solver.lower()]
        except KeyError:
            raise ValueError(f"Error: backend {solver} not supported. Available built-in options: {list(available_mp2_solvers.keys())}")
    elif not issubclass(solver, ElectronicStructureSolver):
        raise TypeError(f"Target must be a str or a subclass of ElectronicStructureSolver but received class {type(solver).__name__}")

    return solver(molecule, **solver_kwargs)


class MP2Solver(ElectronicStructureSolver):
    """Uses the MP2 method to solve the electronic structure problem.

    Args:
        molecule (SecondQuantizedMolecule) : Molecule
        solver (string or Type[ElectronicStructureSolver] or None): Supported string identifiers can be found in
            available_mp2_solvers (see mp2_solver.py). Can also provide a user-defined MP2 implementation
            (child to ElectronicStructureSolver class)
        solver_kwargs: Other arguments that could be passed to a target. Examples are solver type (e.g. dfmp2, mp2), Convergence options etc.

    Attributes:
        solver (Type[ElectronicStructureSolver]): The solver that is used for obtaining the MP2 solution.
     """
    def __init__(self, molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_mp2_solver, **solver_kwargs):
        self.solver = get_mp2_solver(molecule, solver, **solver_kwargs)

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total MP2 energy.
        """
        return self.solver.simulate()

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.
        """
        return self.solver.get_rdm()

    def get_mp2_amplitudes(self):
        """Compute the double amplitudes from the MP2 perturbative method, and
        then reorder the elements into a dense list. The single (T1) amplitudes
        are set to a small non-zero value. The ordering is single, double
        (diagonal), double (non-diagonal).

        Returns:
            list of float: The electronic excitation amplitudes.
        """
        return self.solver.get_mp2_amplitudes()
