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

"""Fragment class, used in construction of ONIOM layers -- contains details of
both the constituent geometry (i.e. which atoms from system are in fragment,
which bonds are broken and how to fix them) as well as the solver(s) to use.
"""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

from tangelo import SecondQuantizedMolecule
from tangelo.algorithms import CCSDSolver, FCISolver, VQESolver, MINDO3Solver, ADAPTSolver, QITESolver
from tangelo.problem_decomposition.oniom._helpers.capping_groups import elements, chemical_groups


class Fragment:

    def __init__(self, solver_low, options_low=None, solver_high=None, options_high=None, selected_atoms=None, charge=0, spin=0, broken_links=None):
        """Fragment class for the ONIOM solver. Each fragment can have broken
        links. In this case, they are capped with a chosen atom. Each fragment
        can also have up to two solvers (low and high accuracy).

        Args:
            solver_low (str): Specification of low accuracy solver for fragment.
            options_low (dict): Specification of low accuracy solver options.
            solver_high (str): Specification of higher accuracy solver for
                fragment.
            options_high (dict): Specification of higher accuracy solver options.
            selected_atoms (list of int or int): Which atoms from molecule are
                in fragment. int counts from start of xyz.
            spin (int): Spin associated witht this fragment.
            charge (int): Charge associated witht this fragment.
            broken_links (list of Link): Bonds broken when forming fragment.

        Attributes:
            mol_low (SecondQuantizedMolecule): Molecule of this fragment, to be
                solved with low accuracy.
            mol_high (SecondQuantizedMolecule): Molecule of this fragment, to be
                solved with high accuracy.
            e_fragment (float): Energy of this fragment as defined by ONIOM.
                None if simulate has not yet been called.
        """

        default_solver_options = {"basis": "sto-3g"}

        # Check to see if a fragment has no solver_high when only a portion of a molecule
        # is selected. If this is allowed, the energy of a fragment is added to the
        # system (all system with solver_low + fragment with solver_low), adding
        # more atoms in the system than there are.
        if selected_atoms is not None and solver_high is None:
            raise RuntimeError("If atoms are selected (selected_atoms different than None), a solver_high must be provided.")

        self.selected_atoms = selected_atoms

        # Solver with low accuracy.
        self.solver_low = solver_low.upper()
        self.options_low = options_low if options_low is not None else default_solver_options

        # Solver with higher accuracy.
        self.solver_high = solver_high.upper() if solver_high is not None else solver_high
        self.options_high = options_high if options_high is not None else default_solver_options

        self.supported_classical_solvers = {"HF": None, "CCSD": CCSDSolver, "FCI": FCISolver, "MINDO3": MINDO3Solver}
        self.supported_quantum_solvers = {"VQE": VQESolver, "ADAPT": ADAPTSolver, "QITE": QITESolver}

        # Check if the solvers are implemented in ONIOM.
        builtin_solvers = self.supported_classical_solvers.keys() | self.supported_quantum_solvers.keys()
        if self.solver_low not in builtin_solvers:
            raise NotImplementedError(f"This {self.solver_low} solver has not been implemented yet in {self.__class__.__name__}")
        elif self.solver_high and self.solver_high not in builtin_solvers:
            raise NotImplementedError(f"This {self.solver_high} solver has not been implemented yet in {self.__class__.__name__}")

        # For this fragment (not the whole molecule).
        self.spin = spin
        self.charge = charge

        self.broken_links = broken_links
        self.mol_low = None
        self.mol_high = None
        self.e_fragment = None

    def build(self):
        """Get the solver objects for this layer. Also defined molecule objects."""

        # Low accuracy solver.
        # We begin by defining the molecule.
        if self.mol_low is None:
            self.mol_low = self.get_mol(self.options_low["basis"], self.options_low.get("frozen_orbitals", None))
            # Basis is only relevant when computing the mean-field. After this,
            # it is discarded (not needed for electronic solver because they
            # retrieved it from the molecule object).
            self.options_low = {i: self.options_low[i] for i in self.options_low if i not in ["basis", "frozen_orbitals"]}

        self.solver_low = self.get_solver(self.mol_low, self.solver_low, self.options_low)

        # Higher accuracy solver.
        if self.solver_high is not None:

            # Molecule is reconstructed (in case a different basis is used).
            if self.mol_high is None:
                self.mol_high = self.get_mol(self.options_high["basis"], self.options_high.get("frozen_orbitals", None))
                # Same process done as in low accuracy process.
                self.options_high = {i: self.options_high[i] for i in self.options_high if i not in ["basis", "frozen_orbitals"]}

            self.solver_high = self.get_solver(self.mol_high, self.solver_high, self.options_high)

    def simulate(self):
        """Get the energy for this fragment.

        Returns:
            float: Energy for the fragment.
        """

        # Low accuracy solver.
        e_low = Fragment.get_energy(self.mol_low, self.solver_low)

        # Higher accuracy solver.
        e_high = 0.
        if self.solver_high is not None:
            e_high = Fragment.get_energy(self.mol_high, self.solver_high)

            # Contribution from low accuracy is substracted, as defined by ONIOM.
            e_low *= -1

        self.e_fragment = e_high + e_low
        return self.e_fragment

    def get_mol(self, basis, frozen=None):
        """Get the molecule object for this fragment (with a specified basis).

        Returns:
            SecondQuantizedMolecule: Molecule object.
        """

        return SecondQuantizedMolecule(self.geometry, self.charge, self.spin, basis, frozen_orbitals=frozen)

    @staticmethod
    def get_energy(molecule, solver):
        """Get the energy for a specific solver.

        Args:
            molecule (SecondQuantizedMolecule): Molecule for this fragment (with
                repaired links).
            solver (solver object or string): Which solver to use.

        Returns:
            float: Energy of the fragment.
        """

        # In case of RHF solver (inside SecondQuantizedMolecule object).
        if isinstance(solver, str):
            energy = molecule.mf_energy
        # The remaining case is for VQESolver, CCSDSolver, FCISolver and
        # MINDO3Solver.
        else:
            energy = solver.simulate()

        return energy

    def get_solver(self, molecule, solver_string, options_solver):
        """Get the solver object (or string for RHF) for this layer.

        Args:
            molecule (SecondQuantizedMolecule): Molecule for this fragment (with
                repaired links).
            solver_string (str): Which solver to use.
            options_solver (dict): Options for the solver.

        Returns:
            ElectronicStructureSolver: Solver object or string.
        """

        if solver_string == "HF":
            return "HF"
        elif solver_string in self.supported_classical_solvers:
            return self.supported_classical_solvers[solver_string](molecule, **options_solver)
        elif solver_string in self.supported_quantum_solvers:
            molecule_options = {"molecule": molecule}
            solver = self.supported_quantum_solvers[solver_string]({**molecule_options, **options_solver})
            solver.build()
            return solver

    def get_resources(self):
        """ Estimate the quantum esources required for this fragment Only
        supports VQESolver solvers.
        """

        # If one of the solver is VQE, quantum resources are returned. If both
        # are VQE, the solver_high overrides the solver_low resources.
        resources = {}

        quantum_solvers = tuple(self.supported_quantum_solvers.values())

        if isinstance(self.solver_low, quantum_solvers):
            resources = self.solver_low.get_resources()

        if isinstance(self.solver_high, quantum_solvers):
            resources = self.solver_high.get_resources()

        return resources


class Link:

    def __init__(self, staying, leaving, factor=1.0, species="H"):
        """Bonds broken during the layer-construction process in ONIOM must be
        mended. This class represents a broken-bond link, and has associated
        methods to generate a new bond, appending the intended species.

        Args:
            staying (int): Atom id retained.
            leaving (int): Atom id lost.
            factor (float) optional: Rescale length of bond, from that in the
                original molecule.
            species (str) optional: Atomic species or a chemical group
                identifier for new link. Can be a list (first element = "X" to
                detect the orientation) for a custom chemical group.
        """

        self.staying = staying
        self.leaving = leaving
        self.factor = factor

        if isinstance(species, str) and species in elements:
            self.species = [(species, (0., 0., 0.))]
        elif isinstance(species, str) and species in chemical_groups:
            self.species = chemical_groups[species]
        elif isinstance(species, (list, tuple)) and species[0][0].upper() == "X":
            self.species = species
        else:
            raise ValueError(f"{species} is not supported. It must be a string identifier or a list of atoms (with a ghost atom ('X') as the first element).")

    def relink(self, geometry):
        """Create atom at location of mended-bond link.

        Args:
            geometry (list of positions): Atomic positions in format
                [[str,tuple(float,float,float)],...].

        Returns:
            list: List of atomic species and position (x, y, z) of replacement
                atom / chemical group.
        """

        elements = [a[0] for a in self.species if a[0].upper() != "X"]
        chem_group_xyz = np.array([[a[1][0], a[1][1], a[1][2]] for a in self.species if a[0].upper() != "X"])

        staying = np.array(geometry[self.staying][1])
        leaving = np.array(geometry[self.leaving][1])

        # Rotation (if not a single atom).
        if len(elements) > 1:
            axis_old = leaving - staying
            axis_new = chem_group_xyz[0] - np.array(self.species[0][1])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rot, _ = R.align_vectors([axis_old], [axis_new])
            chem_group_xyz = rot.apply(chem_group_xyz)

        # Move the atom / group to the right position in space.
        replacement = self.factor*(leaving-staying) + staying
        translation = replacement - chem_group_xyz[0]
        chem_group_xyz += translation

        return [(element, (xyz[0], xyz[1], xyz[2])) for element, xyz in zip(elements, chem_group_xyz)]
