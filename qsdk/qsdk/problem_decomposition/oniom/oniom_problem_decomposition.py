"""Our own n-layered Integrated molecular Orbital and Molecular mechanics (ONIOM) solver.
User specifies either the number (if beginning from start of list), or the indices, of
atoms which are to be identified as the model system(s), from the larger molecular system.

Main model class for running oniom-calculations. This is analogous to the
scf.RHF, etc. methods, requiring however a bit more information. User supplies
an atomic-geometry, and specifies the system, as well as necessary models,
of increasing sophistication.

Reference:
The ONIOM Method and Its Applications
Lung Wa Chung, W. M. C. Sameera, Romain Ramozzi, Alister J. Page, Miho Hatanaka,
Galina P. Petrova, Travis V. Harris, Xin Li, Zhuofeng Ke, Fengyi Liu, Hai-Bei Li,
Lina Ding, and Keiji Morokuma
Chemical Reviews 2015 115 (12), 5678-5796.
DOI: 10.1021/cr5004419
"""
# TODO: Supporting many (3+) layers of different accuracy.
# TODO: Capping with CH3 or other functional groups.

import numpy as np
from pyscf.geomopt.geometric_solver import GeometryOptimizer

from qsdk.problem_decomposition.oniom._helpers.oniom_gradients import ONIOMGradient
from qsdk.problem_decomposition.problem_decomposition import ProblemDecomposition
from qsdk.toolboxes.molecular_computation.molecular_data import atom_string_to_list


class ONIOMProblemDecomposition(ProblemDecomposition):

    def __init__(self, opt_dict):
        """Main class for the ONIOM hybrid solver. It defines layers with
        with different electronic solvers.

        Attributes:
            geometry (strin or list): XYZ atomic coords (in "str float float\n..." or
                [[str, (float, float, float)], ...] format).
            fragments (list of Fragment): Specification of different system-subgroups and
                their solvers.
            verbose (bolean): Verbose flag.
        """

        default_options = {"geometry": None,
                           "fragments": list(),
                           "verbose": False}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}.")

        # Raise error/warnings if input is not as expected
        if not self.geometry or not self.fragments:
            raise ValueError(f"A geometry and models must be provided when instantiating ONIOMProblemDecomposition.")

        self.geometry = atom_string_to_list(self.geometry) if isinstance(self.geometry, str) else self.geometry
        self.update_geometry(self.geometry)

        self.mol = self.fragments[0].mol_low

    def update_geometry(self, new_geometry):
        """For each fragment, the atom selection is passed to the Fragment object.
        Depending on the input, the method has several behaviors. This method is
        compatible with updating the geometry on the fly (optimization).

        Args:
            new_geometry (list): XYZ atomic coords in [[str, (float, float,
                float)], ...].
        """
        self.geometry = new_geometry

        for fragment in self.fragments:
            # Case when no atom are selected -> whole system.
            if fragment.selected_atoms is None:
                fragment_geometry = new_geometry
            # Case where an int is detected -> first n atoms.
            elif type(fragment.selected_atoms) is int:
                fragment_geometry = new_geometry[:fragment.selected_atoms]
            # Case where a list of int is detected -> atom indexes are selected.
            # First atom is 0.
            elif isinstance(fragment.selected_atoms, list) and all(isinstance(id_atom, int) for id_atom in fragment.selected_atoms):
                fragment_geometry = [self.geometry[n] for n in fragment.selected_atoms]
            # Otherwise, an error is raised (list of float, str, etc.).
            else:
                raise TypeError("selected_atoms must be an int or a list of int.")

            # If there are broken_links (other than an empty list nor None).
            # The whole molecule geometry is needed to compute the position of
            # the capping atom (or functional group in the future).
            if fragment.broken_links:
                fragment_geometry += [li.relink(new_geometry) for li in fragment.broken_links]

            fragment.update_geometry(fragment_geometry)

    def simulate(self):
        """Run the ONIOM core-method. The total energy is defined as
        E_ONIOM = E_LOW[SYSTEM] + \sum_i {E_HIGH_i[MODEL_i] - E_LOW_i[MODEL_i]}

        Returns:
            float: Total ONIOM energy.
        """

        # Run energy calculation for each fragment.
        e_oniom = sum([fragment.simulate() for fragment in self.fragments])

        return e_oniom

    def get_jacobian(self, fragment):
        """Get Jacobian, computed for layer-atomic positions, relative to system atomic-positions
        Used in calculation of method gradient
        *return*:
            - **Jmatrix**: numpy array of len(layer) x len(system) float
        """

        Nall = len(self.geometry)
        Natoms = len(fragment.geometry)
        Jmatrix = np.eye(Natoms, Nall)

        # TODO replace this portion of the code.
        try:
            Nlinks = len(fragment.broken_links)
        except TypeError:
            return Jmatrix # When it is None?

        rows = Natoms - (1+ np.mod(np.linspace(0, 2*Nlinks-1, 2*Nlinks, dtype=int), Nlinks))
        cols = np.array([[li.staying, li.leaving] for li in fragment.broken_links]).astype(int).flatten(order='F')
        indices = (rows, cols)

        Jmatrix[(Natoms-Nlinks):, :] = 0.0
        Jmatrix[indices] = np.array([li.factor for li in fragment.broken_links] + [1-li.factor for li in fragment.broken_links])

        return Jmatrix

    def nuc_grad_method(self):
        """Get ONIOM gradient object, from grad module.
        Instantiates class
        *return*:
            - **grad.oniom_grad(self)**: instance of grad.oniom_grad class
        """
        return ONIOMGradient(self)

    run = simulate
    kernel = simulate

    def optimize(self, constraints=None, params=None):
        """
        Run geomeTRIC optimizer backend, applying the oniom solver
        as our method.
        TODO: match Ang, Bohr inputs for successful execution with
        either unit input

        *kwargs*:
            - **constraints**: string, textfile with constraints for optimization
            - **params**: dictionary of parameters for convergence
        *return*:
            - **opt**: optimizer object, containing its mol attribute
        """

        opt = GeometryOptimizer(self)
        opt.run()
        return opt


if __name__ == "__main__":
    pass
