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

"""QM/MM module where the user specifies either the number (if beginning from start of
list), or the indices, of atoms which are to be identified as the model
system(s), from the larger molecular system.

Main model class for running qmmm-calculations. This is analogous to the
scf.RHF, etc. methods, requiring however a bit more information. User supplies
an atomic geometry, and specifies the system, as well as necessary models,
of increasing sophistication.

Reference:
    - A. Warshel, M. Levitt,
    Theoretical studies of enzymic reactions: Dielectric, electrostatic and steric stabilization of the carbonium ion in the reaction of lysozyme,
    Journal of Molecular Biology,
    Volume 103, Issue 2, 1976, Pages 227-249.
    https://doi.org/10.1016/0022-2836(76)90311-9.
"""
from typing import List, Union, Tuple

from tangelo.problem_decomposition.problem_decomposition import ProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list, get_default_integral_solver, get_integral_solver
from tangelo.toolboxes.molecular_computation.mm_charges_solver import MMChargesSolver, get_mm_package
from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


class QMMMProblemDecomposition(ProblemDecomposition):

    def __init__(self, opt_dict: dict):
        """Main class for the QM/MM hybrid solver. It defines a QM region (to be solved using quantum chemistry
        or the quantum computer) and an MM region solved via a force field.

        The QM region includes the partial charges from a force-field calculation.

        Attributes:
            geometry (string or list): XYZ atomic coords (in "str float float..."
                or [[str, (float, float, float)], ...] format) or a pdb file.
                If a pdb file is used, openMM must be installed.
            charges (Union[List[str], List[Tuple[float, Tuple[float, float, float]]]]): The charges and their positions
                used for the electrostatic embedding of the QM region.
                Each item in the list is either a filename or (charge, (x, y, z))
            qmfragment (Fragment): Specification of the QM region its solvers.
            mmpackage (Union[MMChargesSolver, str]): Either a str identifier for an MMChargesSolver or an MMChargesSolverObject
            integral_solver (IntegralSolver): An unitialized IntegralSolver class that accepts charges as an argument in the form
                as defined above.
            verbose (bolean): Verbose flag.
        """

        copt_dict = opt_dict.copy()
        self.geometry: Union[str, List[Tuple[str, Tuple[float]]]] = copt_dict.pop("geometry", None)
        self.charges: Union[List(str), List[Tuple[float, Tuple[float, float, float]]]] = copt_dict.pop("charges", None)
        self.qmfragment: Fragment = copt_dict.pop("qmfragment", None)
        self.verbose: bool = copt_dict.pop("verbose", False)
        mmpackage: Union[MMChargesSolver, str] = copt_dict.pop("mmpackage", "default")
        self.mmpackage: MMChargesSolver = mmpackage if isinstance(mmpackage, MMChargesSolver) else get_mm_package(mmpackage)
        integral_solver: IntegralSolver = copt_dict.pop("integral_solver", "default")

        self.supported_mm_packages = ["openmm", "rdkit"]

        self.mmcharges: List[Tuple[float, Tuple[float, float, float]]] = []
        self.pdbgeometry = False
        self.qmcharges = None

        if isinstance(self.geometry, str):
            self.pdbgeometry = True
            if self.mmpackage is None:
                raise ModuleNotFoundError(f"Any of {self.supported_mm_packages} is required to use {self.__class__.__name__} when supplying only a pdb file")

            qmcharges, self.geometry = self.mmpackage.get_charges([self.geometry])
            self.qmcharges = [(charge, self.geometry[i][1]) for i, charge in enumerate(qmcharges)]
        if isinstance(self.charges, list) and isinstance(self.charges[0], str):
            charges, geometry = self.mmpackage.get_charges(self.charges)
            self.mmcharges += [(charge, geometry[i][1]) for i, charge in enumerate(charges)]

        if copt_dict.keys():
            raise KeyError(f"Keywords :: {copt_dict.keys()}, not available in {self.__class__.__name__}.")

        # Raise error if geometry or qm fragment were not provided by user

        if not self.geometry or not self.qmfragment:
            raise ValueError(f"A geometry and qm fragment must be provided when instantiating {self.__class__.__name__}.")

        self.geometry = atom_string_to_list(self.geometry) if isinstance(self.geometry, str) else self.geometry

        # Distribute atoms to QM region or add to self.mmcharges for use in IntegralSolver
        self.distribute_atoms()

        # Set integral solver
        if isinstance(integral_solver, str):
            self.integral_solver = (get_default_integral_solver(qmmm=True)(charges=self.mmcharges) if integral_solver == "default" else
                                    get_integral_solver(integral_solver, qmmm=True)(charges=self.mmcharges))
        else:
            self.integral_solver = integral_solver(charges=self.mmcharges)

        self.qmfragment.build(self.integral_solver)

    def distribute_atoms(self):
        """For each fragment, the atom selection is passed to the Fragment
        object. Depending on the input, the method has several behaviors.
        """

        if isinstance(self.charges, list) and isinstance(self.charges[0], tuple):
            self.mmcharges += self.charges

        # Case when no atoms are selected -> whole system. i.e. no added partial charges
        if self.qmfragment.selected_atoms is None:
            self.qmfragment.geometry = self.geometry

        # Case where an int is detected -> first n atoms.
        elif type(self.qmfragment.selected_atoms) is int:
            self.qmfragment.geometry = self.geometry[:self.qmfragment.selected_atoms]
            if self.pdbgeometry:
                self.mmcharges += self.qmcharges[self.qmfragment.selected_atoms:]

        # Case where a list of int is detected -> atom indexes are selected.
        # First atom is 0.
        elif isinstance(self.qmfragment.selected_atoms, list) and all(isinstance(id_atom, int) for id_atom in self.qmfragment.selected_atoms):
            self.qmfragment.geometry = [self.geometry[n] for n in self.qmfragment.selected_atoms]
            if self.pdbgeometry:
                self.mmcharges += [self.qmcharges[n] for n in range(len(self.geometry)) if n not in self.qmfragment.selected_atoms]

        # Otherwise, an error is raised (list of float, str, etc.).
        else:
            raise TypeError("selected_atoms must be an int or a list of int.")

        # If there are broken_links (other than an empty list or None).
        # The whole molecule geometry is needed to compute the position of
        # the capping atom (or functional group).
        if self.qmfragment.broken_links:
            for li in self.qmfragment.broken_links:
                self.qmfragment.geometry += li.relink(self.geometry)

    def simulate(self):
        """Run the QM/MM electrostatic embedding calculation

        Returns:
            float: Total QM/MM energy.
        """

        return self.qmfragment.simulate()

    def get_resources(self):
        """Estimate the resources required by QM/MM. Only supports fragments
        solved with quantum solvers. Resources for each fragments are outputed
        as a dictionary.
        """

        quantum_resources = self.qmfragment.get_resources()

        if self.verbose:
            print(f"\t\t{quantum_resources}\n")

        return quantum_resources
