"""Our own n-layered Integrated molecular Orbital and Molecular mechanics
(ONIOM) solver. User specifies either the number (if beginning from start of
list), or the indices, of atoms which are to be identified as the model
system(s), from the larger molecular system.

Main model class for running oniom-calculations. This is analogous to the
scf.RHF, etc. methods, requiring however a bit more information. User supplies
an atomic-geometry, and specifies the system, as well as necessary models,
of increasing sophistication.

Reference:
    - The ONIOM Method and Its Applications. Lung Wa Chung, W. M. C. Sameera,
    Romain Ramozzi, Alister J. Page, Miho Hatanaka, Galina P. Petrova,
    Travis V. Harris, Xin Li, Zhuofeng Ke, Fengyi Liu, Hai-Bei Li, Lina Ding
    and Keiji Morokuma
    Chemical Reviews 2015 115 (12), 5678-5796. DOI: 10.1021/cr5004419.
"""
# TODO: Supporting many (3+) layers of different accuracy.
# TODO: Capping with CH3 or other functional groups.

from qsdk.problem_decomposition.problem_decomposition import ProblemDecomposition
from qsdk.toolboxes.molecular_computation.molecule import atom_string_to_list


class ONIOMProblemDecomposition(ProblemDecomposition):

    def __init__(self, opt_dict):
        """Main class for the ONIOM hybrid solver. It defines layers with
        different electronic solvers.

        Attributes:
            geometry (strin or list): XYZ atomic coords (in "str float float..."
                or [[str, (float, float, float)], ...] format).
            fragments (list of Fragment): Specification of different
                system-subgroups and their solvers.
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
        self.distribute_atoms()

        for frag in self.fragments:
            frag.build()

    def distribute_atoms(self):
        """For each fragment, the atom selection is passed to the Fragment
        object. Depending on the input, the method has several behaviors.
        """

        for fragment in self.fragments:
            # Case when no atom are selected -> whole system.
            if fragment.selected_atoms is None:
                fragment.geometry = self.geometry
            # Case where an int is detected -> first n atoms.
            elif type(fragment.selected_atoms) is int:
                fragment.geometry = self.geometry[:fragment.selected_atoms]
            # Case where a list of int is detected -> atom indexes are selected.
            # First atom is 0.
            elif isinstance(fragment.selected_atoms, list) and all(isinstance(id_atom, int) for id_atom in fragment.selected_atoms):
                fragment.geometry = [self.geometry[n] for n in fragment.selected_atoms]
            # Otherwise, an error is raised (list of float, str, etc.).
            else:
                raise TypeError("selected_atoms must be an int or a list of int.")

            # If there are broken_links (other than an empty list nor None).
            # The whole molecule geometry is needed to compute the position of
            # the capping atom (or functional group in the future).
            if fragment.broken_links:
                fragment.geometry += [li.relink(self.geometry) for li in fragment.broken_links]

    def simulate(self):
        r"""Run the ONIOM core-method. The total energy is defined as
        E_ONIOM = E_LOW[SYSTEM] + \sum_i {E_HIGH_i[MODEL_i] - E_LOW_i[MODEL_i]}

        Returns:
            float: Total ONIOM energy.
        """

        # Run energy calculation for each fragment.
        e_oniom = sum([fragment.simulate() for fragment in self.fragments])

        return e_oniom

    def get_resources(self):
        """Estimate the resources required by ONIOM. Only supports fragments
        solved with VQESolver. Resources for each fragments are outputed as a
        list.
        """

        quantum_resources = [None] * len(self.fragments)

        for fragment_i, fragment in enumerate(self.fragments):
            quantum_resources[fragment_i] = fragment.get_resources()

            if self.verbose:
                if not quantum_resources[fragment_i]:
                    verbose_output = "\t\tRessources estimation not supported for classical solvers."
                else:
                    verbose_output = f"\t\t{quantum_resources[fragment_i]}"

                print(f"\t\tFragment Number : # {fragment_i + 1} \n\t\t{'-'*24}")
                print(f"{verbose_output}\n")

        return quantum_resources