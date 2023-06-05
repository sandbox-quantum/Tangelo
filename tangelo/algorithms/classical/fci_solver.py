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

"""Define electronic structure solver employing the full configuration
interaction (CI) method.
"""
from typing import Union, Type
import itertools

import numpy as np

from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.algorithms.electronic_structure_solver import ElectronicStructureSolver
from tangelo.helpers.utils import installed_chem_backends, deprecated, is_package_installed

if 'pyscf' in installed_chem_backends:
    default_fci_solver = 'pyscf'
elif 'psi4' in installed_chem_backends:
    default_fci_solver = 'psi4'
else:
    default_fci_solver = None


class FCISolverPySCF(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem,
    through pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        mean_field (pyscf.scf): Mean field object.
    """

    def __init__(self, molecule):
        if not is_package_installed("pyscf"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")

        if molecule.uhf:
            raise NotImplementedError(f"SecondQuantizedMolecule that use UHF are not currently supported in {self.__class__.__name__}. Use CCSDSolver")

        from pyscf import ao2mo, fci, mcscf
        self.ao2mo = ao2mo

        self.ci = None
        self.norb = molecule.n_active_mos
        self.nelec = molecule.n_active_electrons
        self.spin = molecule.spin
        self.n_alpha = self.nelec//2 + self.spin//2 + (self.nelec % 2)
        self.n_beta = self.nelec//2 - self.spin//2

        # Need to use a CAS method if frozen orbitals are defined
        if molecule.frozen_mos is not None:
            # Generate CAS space with given frozen_mos, then use pyscf functionality to
            # obtain effective Hamiltonian with frozen orbitals excluded from the CI space.
            self.cas = True
            self.cassolver = mcscf.CASSCF(molecule.mean_field,
                                          molecule.n_active_mos,
                                          (self.n_alpha, self.n_beta))
            mos = self.cassolver.sort_mo([i+1 for i in molecule.active_mos])
            self.h1e_cas, self.ecore = self.cassolver.get_h1eff(mos)
            self.h2e_cas = self.cassolver.get_h2eff(mos)
            # Initialize the FCI solver that will use the effective Hamiltonian generated from CAS
            self.cisolver = fci.direct_spin1.FCI()
        else:
            self.cas = False
            if self.spin == 0:
                self.cisolver = fci.direct_spin0.FCI(molecule.mean_field.mol)
            else:
                self.cisolver = fci.direct_spin1.FCI()

        self.cisolver.verbose = 0
        self.mean_field = molecule.mean_field

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total FCI energy.
        """

        if self.cas:  # Use previously generated effective Hamiltonian to obtain FCI solution
            energy, self.ci = self.cisolver.kernel(self.h1e_cas,
                                                   self.h2e_cas,
                                                   self.norb,
                                                   (self.n_alpha, self.n_beta),
                                                   ecore=self.ecore)
        else:  # Generate full Hamiltonian and obtain FCI solution.
            h1 = self.mean_field.mo_coeff.T @ self.mean_field.get_hcore() @ self.mean_field.mo_coeff

            twoint = self.mean_field._eri

            eri = self.ao2mo.restore(8, twoint, self.norb)
            eri = self.ao2mo.incore.full(eri, self.mean_field.mo_coeff)
            eri = self.ao2mo.restore(1, eri, self.norb)

            ecore = self.mean_field.energy_nuc()

            if self.spin == 0:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], self.nelec, ecore=ecore)
            else:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], (self.n_alpha, self.n_beta), ecore=ecore)

        return energy

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If method "simulate" hasn't been run.
        """

        # Check if Full CI is performed
        if self.ci is None:
            raise RuntimeError("FCISolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        if self.cas:
            one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))
        else:
            if self.spin == 0:
                one_rdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
                two_rdm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)
            else:
                one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))

        return one_rdm, two_rdm


class FCISolverPsi4(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem,
    through pyscf.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        mean_field (pyscf.scf): Mean field object.
    """

    def __init__(self, molecule):
        if not is_package_installed("psi4"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")

        if molecule.uhf:
            raise NotImplementedError(f"SecondQuantizedMolecule that use UHF are not currently supported in {self.__class__.__name__}. Use CCSDSolver")

        from pyscf import ao2mo, fci, mcscf
        self.ao2mo = ao2mo

        self.ci = None
        self.norb = molecule.n_active_mos
        self.nelec = molecule.n_active_electrons
        self.spin = molecule.spin
        self.n_alpha = self.nelec//2 + self.spin//2 + (self.nelec % 2)
        self.n_beta = self.nelec//2 - self.spin//2

        # Need to use a CAS method if frozen orbitals are defined
        if molecule.frozen_mos is not None:
            # Generate CAS space with given frozen_mos, then use pyscf functionality to
            # obtain effective Hamiltonian with frozen orbitals excluded from the CI space.
            self.cas = True
            self.cassolver = mcscf.CASSCF(molecule.mean_field,
                                          molecule.n_active_mos,
                                          (self.n_alpha, self.n_beta))
            mos = self.cassolver.sort_mo([i+1 for i in molecule.active_mos])
            self.h1e_cas, self.ecore = self.cassolver.get_h1eff(mos)
            self.h2e_cas = self.cassolver.get_h2eff(mos)
            # Initialize the FCI solver that will use the effective Hamiltonian generated from CAS
            self.cisolver = fci.direct_spin1.FCI()
        else:
            self.cas = False
            if self.spin == 0:
                self.cisolver = fci.direct_spin0.FCI(molecule.mean_field.mol)
            else:
                self.cisolver = fci.direct_spin1.FCI()

        self.cisolver.verbose = 0
        self.mean_field = molecule.mean_field

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total FCI energy.
        """

        if self.cas:  # Use previously generated effective Hamiltonian to obtain FCI solution
            energy, self.ci = self.cisolver.kernel(self.h1e_cas,
                                                   self.h2e_cas,
                                                   self.norb,
                                                   (self.n_alpha, self.n_beta),
                                                   ecore=self.ecore)
        else:  # Generate full Hamiltonian and obtain FCI solution.
            h1 = self.mean_field.mo_coeff.T @ self.mean_field.get_hcore() @ self.mean_field.mo_coeff

            twoint = self.mean_field._eri

            eri = self.ao2mo.restore(8, twoint, self.norb)
            eri = self.ao2mo.incore.full(eri, self.mean_field.mo_coeff)
            eri = self.ao2mo.restore(1, eri, self.norb)

            ecore = self.mean_field.energy_nuc()

            if self.spin == 0:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], self.nelec, ecore=ecore)
            else:
                energy, self.ci = self.cisolver.kernel(h1, eri, h1.shape[1], (self.n_alpha, self.n_beta), ecore=ecore)

        return energy

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If method "simulate" hasn't been run.
        """

        # Check if Full CI is performed
        if self.ci is None:
            raise RuntimeError("FCISolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        if self.cas:
            one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))
        else:
            if self.spin == 0:
                one_rdm = self.cisolver.make_rdm1(self.ci, self.norb, self.nelec)
                two_rdm = self.cisolver.make_rdm2(self.ci, self.norb, self.nelec)
            else:
                one_rdm, two_rdm = self.cisolver.make_rdm12(self.ci, self.norb, (self.n_alpha, self.n_beta))

        return one_rdm, two_rdm


class FCISolverPsi4(ElectronicStructureSolver):
    """ Uses the Full CI method to solve the electronic structure problem,
    through psi4.

    Args:
        molecule (SecondQuantizedMolecule): The molecule to simulate.

    Attributes:
        ci (numpy.array): The CI wavefunction (float64).
        norb (int): The number of molecular orbitals.
        nelec (int): The number of electrons.
        cisolver (pyscf.fci.direct_spin0.FCI): The Full CI object.
        mean_field (pyscf.scf): Mean field object.
    """

    def __init__(self, molecule: SecondQuantizedMolecule):
        if not is_package_installed("psi4"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package")

        if molecule.uhf:
            raise NotImplementedError(f"SecondQuantizedMolecule that use UHF are not currently supported in {self.__class__.__name__}. Use CCSDSolver")

        import psi4
        self.backend = psi4
        self.molecule = molecule
        self.reference = "rohf" if molecule.spin != 0 else "rhf"
        self.backend.core.set_output_file('output.dat', False)
        if isinstance(molecule.xyz, list):
            input_string = f"{molecule.q} {molecule.spin + 1} \n"
            for line in molecule.xyz:
                input_string += f"{line[0]} {line[1][0]} {line[1][1]} {line[1][2]} \n"
            input_string += "symmetry c1"
            self.psi4molecule = self.backend.geometry(input_string)
        else:
            self.psi4molecule = self.backend.geometry(molecule.xyz)
        self.basis = molecule.basis

    def simulate(self):
        """Perform the simulation (energy calculation) for the molecule.

        Returns:
            float: Total FCI energy.
        """
        self.backend.set_options({'basis': self.basis, 'reference': self.reference, 'opdm': True, 'tpdm': True, 'frozen_docc': [0], 'frozen_uocc': [1]})
        self.molecule.solver.wfn.Ca().rotate_columns(0, 6, 7, np.deg2rad(90))
        #self.molecule.solver.wfn.Cb().rotate_columns(0, 6, 7, np.deg2rad(90))
        energy, self.wfn = self.backend.energy('casscf', molecule=self.psi4molecule, basis=self.basis, return_wfn=True, ref_wfn=self.molecule.solver.wfn)
        self.mints = self.backend.core.MintsHelper(self.wfn.basisset())

        return energy

    def get_rdm(self):
        """Compute the Full CI 1- and 2-particle reduced density matrices.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If method "simulate" hasn't been run.
        """

        # Check if Full CI is performed
        if self.wfn is None:
            raise RuntimeError("FCISolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        one_rdm = self.wfn.get_opdm(0, 0, "SUM", False)
        two_rdm = self.wfn.get_tpdm("SUM", True)

        return one_rdm, two_rdm


fci_solver_dict = {"pyscf": FCISolverPySCF, 'psi4': FCISolverPsi4}


def get_fci_solver(molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_fci_solver, **kwargs):
    """Return requested target backend object.

    Args:
        solver (string or Type[ElectronicStructureSolver] or None): Supported string identifiers can be found in
            fci_solver_dict (from tangelo.algorithms.classical.fci_solver). Can also provide a user-defined backend (child to ElectronicStructureSolver class)
        kwargs: Other arguments that could be passed to a target. Examples are qubits_to_use for a QPU, transpiler
            optimization level, error mitigation flags etc.
     """

    if solver is None and default_fci_solver is not None:
        solver = fci_solver_dict[default_fci_solver]
    # If target is a string use target_dict to return built-in backend
    elif isinstance(solver, str):
        try:
            solver = fci_solver_dict[solver]
        except KeyError:
            raise ValueError(f"Error: backend {solver} not supported. Available built-in options: {list(fci_solver_dict.keys())}")
    elif not issubclass(solver, ElectronicStructureSolver):
        raise TypeError(f"Target must be a str or a subclass of Backend but received class {type(solver).__name__}")

    return solver(molecule, **kwargs)


@deprecated("Please use get_fci_solver.")
def FCISolver(molecule: SecondQuantizedMolecule, solver: Union[None, str, Type[ElectronicStructureSolver]] = default_fci_solver, **kwargs):
    return get_fci_solver(molecule, solver, **kwargs)
