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

import numpy as np

# Imports of electronic solvers and data structure
from tangelo.algorithms import CCSDSolver, FCISolver, VQESolver, MINDO3Solver
from tangelo import SecondQuantizedMolecule


class Fragment:

    def __init__(self, solver_low, options_low=None, solver_high=None, options_high=None, selected_atoms=None, charge=0, spin=0, broken_links=None):
        """Fragment class for the ONIOM solver. Each fragment can have broken
        links. In this case, they are capped with a chosen atom. Each fragment
        can also have up to two solvers (low and high accuracy).

        Args:
            solver_low (str): Specification of low accuracy solver for fragment.
            options_low (str): Specification of low accuracy solver options.
            solver_high (str): Specification of higher accuracy solver for
                fragment.
            options_high (str): Specification of higher accuracy solver options.
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
            self.options_low = {i:self.options_low[i] for i in self.options_low if i not in ["basis", "frozen_orbitals"]}

        self.solver_low = self.get_solver(self.mol_low, self.solver_low, self.options_low)

        # Higher accuracy solver.
        if self.solver_high is not None:

            # Molecule is reconstructed (in case a different basis is used).
            if self.mol_high is None:
                self.mol_high = self.get_mol(self.options_high["basis"], self.options_high.get("frozen_orbitals", None))
                # Same process done as in low accuracy process.
                self.options_high = {i:self.options_high[i] for i in self.options_high if i not in ["basis", "frozen_orbitals"]}

            self.solver_high = self.get_solver(self.mol_high, self.solver_high, self.options_high)

    def simulate(self):
        """Get the energy for this fragment.

        Returns:
            float: Energy for the fragment.
        """

        # Low accuracy solver.
        e_low = self.get_energy(self.mol_low, self.solver_low)

        # Higher accuracy solver.
        e_high = 0.
        if self.solver_high is not None:
            e_high = self.get_energy(self.mol_high, self.solver_high)

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

    def get_energy(self, molecule, solver):
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

        if solver_string == "RHF":
            return "RHF"
        elif solver_string == "CCSD":
            return CCSDSolver(molecule, **options_solver)
        elif solver_string == "FCI":
            return FCISolver(molecule, **options_solver)
        elif solver_string == "MINDO3":
            return MINDO3Solver(molecule, **options_solver)
        elif solver_string == "VQE":
            molecule_options = {"molecule": molecule}
            solver = VQESolver({**molecule_options, **options_solver})
            solver.build()
            return solver
        else:
            raise NotImplementedError(f"This {solver_string} solver has not been implemented yet in ONIOMProblemDecomposition")

    def get_resources(self):
        """ Estimate the quantum esources required for this fragment Only
        supports VQESolver solvers.
        """

        # If one of the solver is VQE, quantum resources are returned. If both
        # are VQE, the solver_high overrides the solver_low resources.
        resources = {}

        if isinstance(self.solver_low, VQESolver):
            resources = self.solver_low.get_resources()

        if isinstance(self.solver_high, VQESolver):
            resources = self.solver_high.get_resources()

        return resources


class Link:

    def __init__(self, staying, leaving, factor=1.0, species="H"):
        """Bonds broken during the layer-construction process in ONIOM must be
        mended. This class represents a broken-bond link, and has associated
        methods to generate a new bond, appending the intended species.

        Args:
            index1 (int): Order in the molecular geometry of atom retained in
                model-unit.
            leaving (int): Order in mol. Geometry of atom lost.
            factor (float) optional: Rescale length of bond, from that in the
                original molecule.
            species (str) optional: Atomic species of appended atom for new
                link.
        """

        self.staying = staying
        self.leaving = leaving
        self.factor = factor
        self.species = species

    def relink(self, geometry):
        """Create atom at location of mended-bond link.

        Args:
            geometry (list of positions): Atomic positions in format
                [[str,tuple(float,float,float)],...].

        Returns:
            str: Atomic species.
            tuple: Position (x, y, z) of replacement atom.
        """

        staying = np.array(geometry[self.staying][1])
        leaving = np.array(geometry[self.leaving][1])
        replacement = self.factor*(leaving-staying) + staying

        return (self.species, (replacement[0], replacement[1], replacement[2]))
