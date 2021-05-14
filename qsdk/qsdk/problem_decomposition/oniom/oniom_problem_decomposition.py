"""Hybrid electronic-structure solver. User specifies either the number (if beginning from start of list), or
the indices, of atoms which are to be identified as the model system(s), from the larger molecular system.

Main model class for running oniom-calculations. This is analogous to the
scf.RHF, etc. methods, requiring however a bit more information. User supplies
an atomic-geometry, and specifies the system, as well as necessary models,
of increasing sophistication.
"""
# TODO: Supporting many (3+) layers of different accuracy.
# TODO:

from qsdk.problem_decomposition.problem_decomposition import ProblemDecomposition
from qsdk.toolboxes.molecular_computation.molecular_data import atom_string_to_list
from qsdk.problem_decomposition.problem_decomposition import ProblemDecomposition


class ONIOMProblemDecomposition(ProblemDecomposition):

    def __init__(self, opt_dict):
        """Main class for the ONIOM hybrid solver. At the moment, it is only
        supporting two layers (high and low accuracy). This can be generalized
        to many layers.

        Attributes:
            geometry (strin or list): XYZ atomic coords (in "str float float\n..." or
                [[str, (float, float, float)], ...] format).
            models (list of Fragment): Specification of different system-subgroups and
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
                raise KeyError(f"Keyword :: {k}, not available in ONIOMProblemDecomposition.")

        # Raise error/warnings if input is not as expected
        if not self.geometry or not self.fragments:
            raise ValueError(f"A geometry and models must be provided when instantiating ONIOMProblemDecomposition.")

        # Converting a single Fragment instance into a list with one element.
        #if isinstance(self.model, Fragment):
        #    self.model = [self.model]

        self.geometry = atom_string_to_list(self.geometry) if isinstance(self.geometry, str) else self.geometry
        self.distribute_atoms()

    def distribute_atoms(self):
        """For each fragment, the atom selection is passed to the Fragment object.
        Depending on the input, the method has several behaviors.
        It calss the Fragment.set_geometry method.
        """

        for fragment in self.fragments:
            # Case when no atom are selected -> whole system.
            if fragment.selected_atoms is None:
                fragment.set_geometry(self.geometry)
            # Case where an int is detected -> first n atoms.
            elif type(fragment.selected_atoms) is int:
                fragment.set_geometry(self.geometry[:fragment.selected_atoms])
            # Case where a list of int is detected -> atom indexes are selected.
            # First atom is 0.
            elif isinstance(fragment.selected_atoms, list) and all(isinstance(_, int) for _ in fragment.selected_atoms):
                fragment.set_geometry([self.geometry[n] for n in fragment.selected_atoms])
            # Otherwise, an error is raised (list of float, str, etc.).
            else:
                raise TypeError("selected_atoms must be an int or a list of int.")

    def simulate(self):
        """Run the ONIOM core-method. The total energy is defined as
        E_ONIOM = E_LOW[SYSTEM] + \sum_i {E_HIGH_i[MODEL_i] - E_LOW_i[MODEL_i]}

        Returns:
            float: Total energy
        """

        e_oniom = 0.
        # Run energy calculation for each fragment.
        for fragment in self.fragments:
            e_oniom += fragment.simulate()

        return e_oniom
