"""Fragment class, used in construction of ONIOM layers -- contains details of both the
constituent geometry (i.e. which atoms from system are in fragment, which bonds are broken
and how to fix them) as well as the solver(s) to use.
"""

import numpy as np
from pyscf import gto

from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.electronic_structure_solvers.ccsd_solver import CCSDSolver
from qsdk.electronic_structure_solvers.fci_solver import FCISolver
from qsdk.electronic_structure_solvers.vqe_solver import VQESolver


class Fragment:

    def __init__(self, solver_low, options_low=None, solver_high=None, options_high=None, selected_atoms=None, charge=0, spin=0, broken_links=None):
        """Main class for the ONIOM hybrid solver.

        Args:
            solver (list of dict or dict): Specification of solver(s) for fragment.
            select_atoms (list of int or int): Which atoms from molecule are in fragment. int counts from start of xyz.
            links (list of Link): Bonds broken when forming fragment.
            spin (int): Spin associated witht this fragment.
            charge (int): Charge associated witht this fragment.

        Attributes:
            broken_links (list of Link): Broken link in this fragment.
            mol_low (pyscf.gto.Mole): PySCF molecule of this fragment, to be solved with low accuracy.
            mol_high (pyscf.gto.Mole): PySCF molecule of this fragment, to be solved with high accuracy.
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

    def set_geometry(self, geometry=None):
        """Stter for the fragment geometry.

        Args:
            geometry (strin or list): XYZ atomic coords (in "str float float\n..." or
                [[str, (float, float, float)], ...] format).
        """

        self.geometry = geometry

        if self.broken_links:
            self.fix_links(geometry)

    def fix_links(self, geometry):
        """Mend broken links to the system, preserving the electron configuration of the fragment.

        *return*:
            - **geometry**: list of atomic labels, locations in the model fragment
        """

        self.geometry += [li.relink(geometry) for li in self.broken_links]

    def simulate(self):
        """Get the solver object for this layer.

        Returns:
            float: Energy for the fragment.
        """

        # Low accuracy solver.
        # We begin by defining the molecule.
        if self.mol_low is None:
            self.mol_low = self.get_mol(self.options_low["basis"])
            # Basis is only relevant when making the pyscf molecule. After this,
            # it is discarded (not needed for electronic solver becasue they retrieved
            # it from the molecule object).
            self.options_low = {i:self.options_low[i] for i in self.options_low if i!="basis"}

        e_low = self.get_energy(self.mol_low, self.solver_low, self.options_low)

        # Higher accuracy solver.
        e_high = 0.
        if self.solver_high is not None:

            # Molecule is reconstructed (in case a different basis is used).
            if self.mol_high is None:
                self.mol_high = self.get_mol(self.options_high["basis"])
                # Same process done as in low accuracy process.
                self.options_high = {i:self.options_high[i] for i in self.options_high if i!="basis"}

            e_high = self.get_energy(self.mol_high, self.solver_high, self.options_high)

            # Contribution from low accuracy is substracted, as defined by ONIOM.
            e_low *= -1

        self.e_fragment = e_high + e_low
        return self.e_fragment

    def get_mol(self, basis):
        """Get the molecule object for this fragment (with a specified basis).

        Returns:
            pyscf.gto.Mole: Molecule object.
        """

        mol= gto.Mole()
        mol.build(atom=self.geometry,
                  basis=basis,
                  charge=self.charge,
                  spin=self.spin)

        return mol

    def get_energy(self, molecule, solver, options_solver):
        """Get the solver object for this layer.

        Args:
            molecule (pyscf.gto): Molecule for this fragment (with repaired links).
            sovler (str): Which solver to use.
            options_solver (dict): Options for the solver.

        Returns:
            float: Energy of the fragment.
        """

        if solver == "RHF":
            mean_field = prepare_mf_RHF(molecule, **options_solver)
            energy = mean_field.e_tot
        elif solver == "CCSD":
            solver = CCSDSolver()
            energy = solver.simulate(molecule, **options_solver)
        elif solver == "FCI":
            solver = FCISolver()
            energy = solver.simulate(molecule, **options_solver)
        elif solver == "VQE":
            molecule_options = {'molecule': molecule}
            solver = VQESolver({**molecule_options, **options_solver})
            solver.build()
            energy = solver.simulate()
        else:
            raise NotImplementedError("This {} solver has not been implemented yet in ONIOMProblemDecomposition".format(solver))

        return energy


class Link:

    def __init__(self, staying, leaving, factor=1.0, species='H'):
        """Bonds broken during the layer-construction process in ONIOM must be mended.
        This class represents a broken-bond link, and has associated methods to generate
        a new bond, appending the intended species.

        Args:
            index1 (int): Order in the molecular geometry of atom retained in model-unit.
            leaving (int): Order in mol. Geometry of atom lost.
            factor (float) optional: Rescale length of bond, from that in the original molecule.
            species (str) optional: Atomic species of appended atom for new link.
        """

        self.staying = staying
        self.leaving = leaving
        self.factor = factor
        self.species = species

    def relink(self, geometry):
        """Create atom at location of mended-bond link.

        Args:
            geometry (list of positions): Atomic positions in format [[str,tuple(float,float,float)],...].

        Returns:
            Atomic species, and position of replacement atom.
        """

        staying = np.array(geometry[self.staying][1])
        leaving = np.array(geometry[self.leaving][1])
        replacement = self.factor*(leaving-staying) + staying

        return [self.species, (replacement[0], replacement[1], replacement[2])]
