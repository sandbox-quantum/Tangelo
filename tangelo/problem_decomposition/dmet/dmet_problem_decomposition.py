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

"""Employ DMET as a problem decomposition technique."""

from enum import Enum
from typing import Union, List, Callable, Dict

import numpy as np
import scipy
import warnings

from tangelo.helpers.utils import is_package_installed
from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition.dmet import _helpers as helpers
from tangelo.problem_decomposition.problem_decomposition import ProblemDecomposition
from tangelo.problem_decomposition.electron_localization import iao_localization, meta_lowdin_localization, nao_localization
from tangelo.algorithms import FCISolver, CCSDSolver, VQESolver
from tangelo.toolboxes.post_processing.mc_weeny_rdm_purification import mcweeny_purify_2rdm
from tangelo.toolboxes.molecular_computation.rdms import pad_rdms_with_frozen_orbitals_restricted, \
    pad_rdms_with_frozen_orbitals_unrestricted
from tangelo.toolboxes.molecular_computation.integral_solver_pyscf import mol_to_pyscf


class Localization(Enum):
    """Enumeration of the electron localization supported by DMET."""
    meta_lowdin = 0
    iao = 1
    nao = 2


class DMETProblemDecomposition(ProblemDecomposition):
    """DMET single-shot algorithm is used for problem decomposition technique.
    By default, CCSD is used as the electronic structure solver, and Meta-Lowdin
    is used for the localization scheme. Users can define other electronic
    structure solver such as FCI or VQE as an impurity solver. IAO scheme can be
    used instead of the Meta-Lowdin localization scheme, but it cannot be used
    for minimal basis set.

    The underlying mean-field for the computation is automatically detected
    from the SecondQuantizedMolecule. RHF, ROHF and UHF mean-fields are
    supported.

    Attributes:
        molecule (SecondQuantizedMolecule): The molecular system.
        electron_localization (Localization): A type of localization scheme.
            Default is Meta-Lowdin.
        fragment_atoms (list): List of number of atoms in each fragment. Sum of
            this list should be the same as the number of atoms in the original
            system.
        fragment_solvers (list or str): List of solvers for each fragment. If
            only a string is detected, this solver is used for all fragments.
        fragment_frozen_orbitals (list): List of frozen orbitals. List of
            integers freezes the n first orbitals in the fragment. Nested list
            of integers freezes orbitals based on indices.
        optimizer (function handle): A function defining the classical optimizer
            and its behavior.
        initial_chemical_potential (float) : Initial value for the chemical
            potential.
        solvers_options (list or dict): List of dictionaries for the solver
            options. If only a single dictionary is passed, the same options are
            applied for every solver. This will raise an error if different
            solvers are parsed.
        virtual_orbital_threshold (float): Occupation threshold for the virtual
            orbital space. Setting it to 0. is the equivalent of turning off
            the virtual orbital space truncation.
            Default=1e-13.
        verbose (bool) : Flag for DMET verbosity.
    """

    def __init__(self, opt_dict):
        if not is_package_installed("pyscf"):
            raise ModuleNotFoundError(f"Using {self.__class__.__name__} requires the installation of the pyscf package.")
        from pyscf import gto, scf
        from tangelo.problem_decomposition.dmet.fragment import SecondQuantizedDMETFragment
        default_ccsd_options = dict()
        default_fci_options = dict()
        default_vqe_options = {"qubit_mapping": "jw",
                               "initial_var_params": "ones",
                               "verbose": False}

        copt_dict = opt_dict.copy()
        self.molecule: SecondQuantizedMolecule = copt_dict.pop("molecule", None)
        self.electron_localization: Localization = copt_dict.pop("electron_localization", Localization.meta_lowdin)
        self.fragment_atoms: List[int] = copt_dict.pop("fragment_atoms", list())
        self.fragment_solvers: Union[str, List[str]] = copt_dict.pop("fragment_solvers", "ccsd")
        self.fragment_frozen_orbitals: List[List[Union[int, str]]] = copt_dict.pop("fragment_frozen_orbitals", list())
        self.optimizer: Callable[..., float] = copt_dict.pop("optimizer", self._default_optimizer)
        self.initial_chemical_potential: float = copt_dict.pop("initial_chemical_potential", 0.0)
        self.solvers_options: List[dict] = copt_dict.pop("solvers_options", list())
        self.virtual_orbital_threshold: float = copt_dict.pop("virtual_orbital_threshold", 1e-13)
        self.verbose: bool = copt_dict.pop("verbose", False)

        self.builtin_localization = set(Localization)

        self.fragment_builder = SecondQuantizedDMETFragment

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        # Raise error/warnings if input is not as expected
        if not self.molecule:
            raise ValueError(f"A SecondQuantizedMolecule object must be provided when instantiating DMETProblemDecomposition.")

        self.uhf = self.molecule.uhf

        # Converting our interface to pyscf.mol.gto and pyscf.scf (used by this
        # code).
        self.mean_field = self.molecule.mean_field
        self.molecule = mol_to_pyscf(self.molecule, self.molecule.basis, ecp=self.molecule.ecp)

        # If fragment_atoms is detected as a nested list of int, atoms are reordered to be
        # consistent with a list of numbers representing the number of atoms in each fragment.
        if isinstance(self.fragment_atoms, list) and all(isinstance(list_atoms, list) for list_atoms in self.fragment_atoms):
            fragment_atoms_flatten = [atom_id for frag in self.fragment_atoms for atom_id in frag]

            if max(fragment_atoms_flatten) >= self.molecule.natm:
                raise RuntimeError("An atom id is higher than the number of atom (indices start at 0).")
            elif len(fragment_atoms_flatten) != len(set(fragment_atoms_flatten)):
                raise RuntimeError("Atom indices must only appear once.")

            # Converting fragment_atoms to an expected list of number of atoms (not atom ids).
            new_fragment_atoms = [len(frag) for frag in self.fragment_atoms]

            # Reordering the molecule geometry.
            new_geometry = [self.molecule._atom[atom_id] for atom_id in fragment_atoms_flatten]

            # Building a new PySCF molecule with correct ordering.
            new_molecule = gto.Mole()
            new_molecule.atom = new_geometry
            new_molecule.basis = self.molecule.basis
            new_molecule.charge = self.molecule.charge
            new_molecule.spin = self.molecule.spin
            new_molecule.unit = "B"
            new_molecule.build()

            # Attribution of the expected fragment_atoms and a reordered molecule.
            self.molecule = new_molecule
            self.fragment_atoms = new_fragment_atoms

            # Force recomputing the mean field if the atom ordering has been changed.
            warnings.warn("The mean field will be recomputed even if one has been provided by the user.", RuntimeWarning)
            self.mean_field = scf.UHF(self.molecule) if self.uhf else scf.RHF(self.molecule)
            self.mean_field.verbose = 0
            self.mean_field.scf()

        # Check if the number of fragment sites is equal to the number of atoms in the molecule
        if self.molecule.natm != sum(self.fragment_atoms):
            raise RuntimeError("The number of fragment sites is not equal to the number of atoms in the molecule")

        if not len(self.fragment_frozen_orbitals):
            self.fragment_frozen_orbitals = [None] * len(self.fragment_atoms)
        elif len(self.fragment_frozen_orbitals) != len(self.fragment_atoms):
            raise RuntimeError("The number of elements in the frozen orbitals list does not match the number of fragments.")

        # Check that the number of solvers matches the number of fragments.
        # If a single string is detected, it is converted to a list.
        # If a list is detected, it must have the same length than the fragment_atoms.
        if isinstance(self.fragment_solvers, str):
            self.fragment_solvers = [self.fragment_solvers for _ in range(len(self.fragment_atoms))]
        elif isinstance(self.fragment_solvers, list):
            if len(self.fragment_solvers) != len(self.fragment_atoms):
                raise RuntimeError("The number of solvers does not match the number of fragments.")

        # Check that the number of solvers options matches the number of solvers.
        # If there is no options, all default ones are applied.
        # If a single options dictionary is parsed, it is repeated for every solvers.
        # If it is a list, we verified that the length is the same.
        if not self.solvers_options:
            for solver in self.fragment_solvers:
                if solver == "ccsd":
                    self.solvers_options.append(default_ccsd_options)
                elif solver == "fci":
                    self.solvers_options.append(default_fci_options)
                elif solver == "vqe":
                    self.solvers_options.append(default_vqe_options)
        elif isinstance(self.solvers_options, dict):
            self.solvers_options = [self.solvers_options for _ in self.fragment_solvers]
        elif isinstance(self.solvers_options, list):
            if len(self.solvers_options) != len(self.fragment_solvers):
                raise RuntimeError("The number of solvers options does not match the number of solvers.")

        # Results of the DMET loops.
        self.chemical_potential = None
        self.dmet_energy = None

        # Define during the building phase (self.build()).
        self.orbitals = None
        self.orb_list = None
        self.orb_list2 = None
        self.onerdm_low = None

        # If save_results in _oneshot_loop is True, the dict is populated.
        self.solver_fragment_dict: Dict[int, VQESolver] = dict()

        # To keep track the number of iteration (was done with an energy list
        # before).
        self.n_iter = 0

    @property
    def quantum_fragments_data(self):
        """This aims to return a dictionary with all necessary components to
        run a quantum experiment for a (or more) DMET fragment(s).
        """

        if not self.solver_fragment_dict:
            raise RuntimeError("Simulate method must be called beforehand.")

        quantum_data = dict()

        # Construction of a dict with SecondQuantizedDMETFragment, qubit
        # hamiltonian and quantum circuit.
        for fragment_i, vqe_object in self.solver_fragment_dict.items():
            quantum_data[fragment_i] = (vqe_object.molecule,
                vqe_object.qubit_hamiltonian, vqe_object.optimal_circuit)

        return quantum_data

    def build(self):
        """Building the orbitals list for each fragment. It sets the values of
        self.orbitals, self.orb_list and self.orb_list2.
        """

        # Locate electron with a built-in method. A custom one can be provided.
        if type(self.electron_localization) == Localization:
            if self.electron_localization == Localization.meta_lowdin:
                self.electron_localization = meta_lowdin_localization
            elif self.electron_localization == Localization.iao:
                self.electron_localization = iao_localization
            elif self.electron_localization == Localization.nao:
                self.electron_localization = nao_localization
            else:
                raise ValueError(f"Unsupported ansatz. Built-in localization methods:\n\t{self.builtin_localization}")
        elif not callable(self.electron_localization):
            raise TypeError(f"Invalid electron localization function. Expecting a function.")

        # Construct orbital object.
        self.orbitals = helpers._orbitals(self.molecule, self.mean_field, range(self.molecule.nao_nr()), self.electron_localization, self.uhf)

        # TODO: remove last argument, combining fragments not supported.
        self.orb_list, self.orb_list2, _ = helpers._fragment_constructor(self.molecule, self.fragment_atoms, 0)

        # Calculate the 1-RDM for the entire molecule.
        if self.molecule.spin == 0 and not self.uhf:
            self.onerdm_low = helpers._low_rdm_rhf(self.orbitals.active_fock, self.orbitals.number_active_electrons)
        else:
            self.onerdm_low = helpers._low_rdm_rohf_uhf(self.orbitals.active_fock_alpha,
                                                        self.orbitals.active_fock_beta,
                                                        self.orbitals.number_active_electrons_alpha,
                                                        self.orbitals.number_active_electrons_beta)

    def simulate(self):
        """Perform DMET loop to optimize the chemical potential. It converges
        when the electron summation across all fragments is the same as the
        number of electron in the molecule.

        Returns:
            float: The DMET energy (dmet_energy).
        """

        # Initialize the energy list and SCF procedure employing newton-raphson algorithm.
        # TODO : find a better initial guess than 0.0 for chemical potential. DMET fails often currently.
        if not self.orbitals:
            raise RuntimeError("No fragment built. Have you called DMET.build ?")

        self.chemical_potential = self.optimizer(self._oneshot_loop, self.initial_chemical_potential)
        self.chemical_potential = self.chemical_potential.real

        # run one more time to save results
        _ = self._oneshot_loop(self.chemical_potential, save_results=True)

        if self.verbose:
            print(" \t*** DMET Cycle Done *** ")
            print(" \tDMET Energy ( a.u. ) = " + "{:17.10f}".format(self.dmet_energy))
            print(" \tChemical Potential   = " + "{:17.10f}".format(self.chemical_potential))

        return self.dmet_energy

    def energy_error_bars(self, n_shots, n_resamples, purify=False, rdm_measurements=None):
        """Perform bootstrapping of measured qubit terms in RDMs to obtain error
        bars for the calculated energy. Can also perform McWeeny purification of
        the RDMs between resampling and calculating the energy.

        Args:
            n_shots (int): The number of shots used to resample from qubit
                terms.
            n_resamples (int): The number of bootstrapping samples for the
                estimate of the energy and standard deviation.
            purify (bool): Use mc_weeny_rdm_purification technique on 2-RDMs.
                Will only apply to fragments with 2 electrons.
            rdm_measurements (dict): A dictionary with keys being the DMET
                fragment number and corresponding values of a dictionary with
                keys of qubit terms in the rdm and corresponding values of a
                frequencies dictionary of the measurements.
                Example: {
                0: {((0, "X"), (1, "Y")): {"10": 0.5, "01": 0.5},
                    ((1, "Z"), (1, "X")): {"00": 0.25, "11": 0.25, "01": 0.5}}
                1: {((0, "Z")): {"0000": 1}
                    ((0, "X"), (1, "Y")): {"1111": 0.5, "0101": 0.5}}
                }.
                If run _oneshot_loop with save_results=True is called:
                self.rdm_measurements will have the proper dictionary format
                with all values populated.

        Returns:
            float: The bootstrapped DMET energy and standard deviation.
        """

        if self.chemical_potential is None:
            raise RuntimeError("No chemical_potential. Have you run a simulation yet?")

        # begin resampling
        resampled_energies = np.zeros(n_resamples, dtype=float)
        for i in range(n_resamples):
            _ = self._oneshot_loop(self.chemical_potential, save_results=False, resample=True,
                                   n_shots=n_shots, purify=purify, rdm_measurements=rdm_measurements)
            resampled_energies[i] = self.dmet_energy.real

        energy_average, energy_standard_deviation = np.mean(resampled_energies), np.std(resampled_energies, ddof=1)

        return energy_average, energy_standard_deviation

    def _build_scf_fragments(self, chemical_potential):
        """Building the orbitals list for each fragment. It sets the values of
        self.orbitals, self.orb_list and self.orb_list2.

        Args:
            onerdm_low (matrix): 1-RDM for the whole molecule.
            chemical_potential (float): Variational parameter for DMET.

        Returns:
            list: List of many objects important for each fragments.
        """

        # Empty list, all fragment informations will be returned in this list.
        scf_fragments = list()

        for i, norb in enumerate(self.orb_list):
            t_list = list()
            t_list.append(norb)
            temp_list = self.orb_list2[i]

            if self.verbose:
                print(f"\tSCF Occupancy Eigenvalues for Fragment Number : # {i}")

            # Construct bath orbitals.
            bath_orb, e_occupied = helpers._fragment_bath(self.orbitals.mol_full, t_list, temp_list,
                                                          self.onerdm_low, self.virtual_orbital_threshold, self.verbose)

            # Obtain one particle rdm for a fragment.
            norb_high, nelec_high, onerdm_high = helpers._fragment_rdm(t_list, bath_orb, e_occupied,
                                                                       self.orbitals.number_active_electrons)

            # Obtain one particle rdm for a fragment.
            one_ele, fock, two_ele = self.orbitals.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

            # Construct guess orbitals for fragment SCF calculations.
            # Carry out SCF calculation for a fragment.
            if self.uhf or self.molecule.spin != 0:
                guess_orbitals, nelec_high_ab = helpers._fragment_guess_rohf_uhf(
                    t_list, bath_orb, chemical_potential, norb_high, nelec_high,
                    self.orbitals.active_fock_alpha, self.orbitals.active_fock_beta,
                    self.orbitals.number_active_electrons_alpha,
                    self.orbitals.number_active_electrons_beta)

                mf_fragment, fock_frag_copy, mol_frag = helpers._fragment_scf_rohf_uhf(
                    nelec_high_ab, two_ele, fock, nelec_high, norb_high,
                    guess_orbitals, chemical_potential, self.uhf)
            else:
                guess_orbitals = helpers._fragment_guess_rhf(t_list, bath_orb, chemical_potential, norb_high, nelec_high,
                                                             self.orbitals.active_fock)
                mf_fragment, fock_frag_copy, mol_frag = helpers._fragment_scf_rhf(
                    t_list, two_ele, fock, nelec_high, norb_high, guess_orbitals,
                    chemical_potential)

            scf_fragments.append([mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock])

        return scf_fragments

    def _oneshot_loop(self, chemical_potential, save_results=False, resample=False, n_shots=None, purify=False, rdm_measurements=None):
        """Perform the DMET loop. This is the cost function which is optimized
        with respect to the chemical potential.

        Args:
            chemical_potential (float): The chemical potential.
            save_results (bool): If True, the VQESolver Class (solver_fragment)
                for each fragment that uses VQE is saved in
                self.solver_fragment_dict with the key being its fragment
                number.
            resample (bool): If True, the saved frequencies are resampled using
                bootstrapping.
            n_shots (int): The number of shots used for resampling, Ideally the
                same as used in the simulation but can be different.
            purify (bool): If True, use McWeeny"s purification technique to
                purify 2-RDM. Only called for fragments with 2 electrons.
            rdm_measurements (dict): A dictionary with keys being the DMET
                fragment number and corresponding values of a dictionary with
                keys of qubit terms in the rdm and corresponding values of a
                frequencies dictionary of the measurements.
                Example: {
                0: {((0, "X"), (1, "Y")): {"10": 0.5, "01": 0.5},
                    ((1, "Z"), (1, "X")): {"00": 0.25, "11": 0.25, "01": 0.5}}
                1: {((0, "Z")): {"0000": 1}
                    ((0, "X"), (1, "Y")): {"1111": 0.5, "0101": 0.5}}
                }.
                If run _oneshot_loop with save_results=True is called:
                self.rdm_measurements will have the proper dictionary format
                with all values populated.

        Returns:
            float: The new chemical potential.
        """
        self.n_iter += 1
        if self.verbose:
            print(" \tIteration = ", self.n_iter)
            print(" \t----------------")
            print(" ")

        number_of_electron = 0.0
        energy_temp = 0.0

        # Possibly add dictionary of measured frequencies for each fragment
        if resample:
            if save_results:
                raise ValueError("Can not save results and resample in same run. Must run saveresults first")
            if not hasattr(self, "solver_fragment_dict"):
                raise AttributeError("Need to run _oneshot_loop with save_results=True in order to resample")
            if rdm_measurements:
                for k, v in rdm_measurements.items():
                    self.solver_fragment_dict[k].rdm_freq_dict = v
            scf_fragments = self.scf_fragments
        else:
            # Carry out SCF calculation for all fragments.
            scf_fragments = self._build_scf_fragments(chemical_potential.real)
        if save_results:
            self.solver_fragment_dict = dict()
            self.rdm_measurements = dict()

        # Iterate across all fragment and compute their energies.
        # The total energy is stored in energy_temp.
        for i, info_fragment in enumerate(scf_fragments):

            # Unpacking the information for the selected fragment.
            mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock = info_fragment

            # Interface with our data strcuture.
            # We create a dummy SecondQuantizedMolecule with a DMETFragment class.
            # It has the same important attributes and methods to be used with
            # functions of this package.
            dummy_mol = self.fragment_builder(mol_frag, mf_fragment, fock,
                fock_frag_copy, t_list, one_ele, two_ele, self.uhf,
                self.fragment_frozen_orbitals[i])

            if self.verbose:
                print("\t\tFragment Number : # ", i + 1)
                print("\t\t------------------------")

            # TODO: Changing this into something more simple is preferable. There
            # would be an enum class with every solver in it. After this, we would
            # define every solver in a list and call them recursively.
            # FCISolver and CCSDSolver must be taken care of, but this is a PR itself.
            solver_fragment = self.fragment_solvers[i]
            solver_options = self.solvers_options[i]
            if solver_fragment == "fci":
                solver_fragment = FCISolver(dummy_mol, **solver_options)
                solver_fragment.simulate()
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == "ccsd":
                solver_fragment = CCSDSolver(dummy_mol, **solver_options)
                solver_fragment.simulate()
                onerdm, twordm = solver_fragment.get_rdm()
            elif solver_fragment == "vqe":
                if resample:
                    solver_fragment = self.solver_fragment_dict[i]
                    if rdm_measurements:
                        if i in rdm_measurements:
                            solver_fragment.rdm_freq_dict = rdm_measurements[i]
                        else:
                            raise KeyError(f"rdm_measurements for fragment {i} are missing")
                    if n_shots:
                        solver_fragment.backend.n_shots = n_shots
                    if solver_fragment.backend.n_shots is None:
                        raise ValueError("n_shots must be specified in original calculation or in error calculation")
                else:
                    system = {"molecule": dummy_mol}
                    solver_fragment = VQESolver({**system, **solver_options})
                    solver_fragment.build()
                    solver_fragment.simulate()

                if purify and solver_fragment.molecule.n_active_electrons == 2 and not self.uhf:
                    onerdm, twordm = solver_fragment.get_rdm(solver_fragment.optimal_var_params, resample=resample, sum_spin=False)
                    onerdm, twordm = mcweeny_purify_2rdm(twordm)
                elif self.uhf:
                    onerdm, twordm = solver_fragment.get_rdm_uhf(solver_fragment.optimal_var_params, resample=resample)
                else:
                    onerdm, twordm = solver_fragment.get_rdm(solver_fragment.optimal_var_params, resample=resample)
                if save_results:
                    self.solver_fragment_dict[i] = solver_fragment
                    self.rdm_measurements[i] = self.solver_fragment_dict[i].rdm_freq_dict

            # Compute the fragment energy and sum up the number of electrons
            if self.uhf:
                onerdm_padded, twordm_padded = pad_rdms_with_frozen_orbitals_unrestricted(dummy_mol, onerdm, twordm)
                fragment_energy, onerdm_a, onerdm_b = self._compute_energy_unrestricted(dummy_mol, onerdm_padded, twordm_padded)
                n_electron_frag = np.trace(onerdm_a[ : t_list[0], : t_list[0]]) + np.trace(onerdm_b[ : t_list[0], : t_list[0]])
            else:
                onerdm_padded, twordm_padded = pad_rdms_with_frozen_orbitals_restricted(dummy_mol, onerdm, twordm)
                fragment_energy, onerdm = self._compute_energy_restricted(dummy_mol, onerdm_padded, twordm_padded)
                n_electron_frag = np.trace(onerdm[: t_list[0], : t_list[0]])

            number_of_electron += n_electron_frag

            # Sum up the energy.
            energy_temp += fragment_energy

            if self.verbose:
                print(f"\t\tFragment Energy = {fragment_energy}\n\t\tNumber of Electrons in Fragment = {n_electron_frag}")

        energy_temp += self.orbitals.core_constant_energy
        self.dmet_energy = energy_temp.real

        if save_results:
            self.scf_fragments = scf_fragments

        return number_of_electron - self.orbitals.number_active_electrons

    def get_resources(self):
        """Estimate the resources required by DMET. Only supports fragments
        solved with VQESolver. Resources for each fragments are outputed as a
        list.
        """

        # Carry out SCF calculation for all fragments.
        scf_fragments = self._build_scf_fragments(self.initial_chemical_potential)

        # Store resources for each fragments.
        resources_fragments = dict()

        # Iterate across all fragment and compute their energies.
        # The total energy is stored in energy_temp.
        for i, info_fragment in enumerate(scf_fragments):

            # Unpacking the information for the selected fragment.
            mf_fragment, fock_frag_copy, mol_frag, t_list, one_ele, two_ele, fock = info_fragment

            dummy_mol = self.fragment_builder(mol_frag, mf_fragment, fock,
                fock_frag_copy, t_list, one_ele, two_ele, self.uhf,
                self.fragment_frozen_orbitals[i])

            # Buiding SCF fragments and quantum circuit. Resources are then
            # estimated. For classical sovlers, this functionality is not
            # implemented yet.
            solver_fragment = self.fragment_solvers[i]
            solver_options = self.solvers_options[i]
            if solver_fragment == "vqe":
                system = {"molecule": dummy_mol}
                solver_fragment = VQESolver({**system, **solver_options})
                solver_fragment.build()
                vqe_resources = solver_fragment.get_resources()
                resources_fragments[i] = vqe_resources

                if self.verbose:
                    print("\t\tFragment Number : # ", i + 1)
                    print("\t\t------------------------")
                    print(f"\t\t{vqe_resources}\n")

        return resources_fragments

    def _compute_energy_restricted(self, dmet_fragment, onerdm, twordm):
        """Calculate the fragment energy.

        Args:
            dmet_fragment (SecondQuantizedDMETFragment): Self-explanatory.
            onerdm (numpy.array): one-particle reduced density matrix (float).
            twordm (numpy.array): two-particle reduced density matrix (float).

        Returns:
            float: Fragment energy (fragment_energy).
            numpy.array: One-particle RDM for a fragment (onerdm, float).
        """

        fock = dmet_fragment.fock
        t_list = dmet_fragment.t_list
        oneint = dmet_fragment.one_ele
        twoint = dmet_fragment.two_ele
        mo_coeff = dmet_fragment.mean_field.mo_coeff

        norb = t_list[0]

        # Calculate the one- and two- RDM for DMET energy calculation (Transform to AO basis)
        onerdm = mo_coeff @ onerdm @ mo_coeff.T

        twordm = np.einsum("pi,ijkl->pjkl", mo_coeff, twordm)
        twordm = np.einsum("qj,pjkl->pqkl", mo_coeff, twordm)
        twordm = np.einsum("rk,pqkl->pqrl", mo_coeff, twordm)
        twordm = np.einsum("sl,pqrl->pqrs", mo_coeff, twordm)

        # Calculate fragment expectation value
        fragment_energy_onerdm = np.einsum("ij,ij->", onerdm[: norb, :], fock[: norb, :] + oneint[: norb, :]) \
            + np.einsum("ij,ij->", onerdm[:, : norb], fock[:, : norb] + oneint[:, : norb])
        fragment_energy_onerdm *= 0.25

        fragment_energy_twordm = np.einsum("ijkl,ijkl->", twordm[: norb, :, :, :], twoint[: norb, :, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm[:, : norb, :, :], twoint[:, : norb, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm[:, :, : norb, :], twoint[:, :, : norb, :]) \
            + np.einsum("ijkl,ijkl->", twordm[:, :, :, : norb], twoint[:, :, :, : norb])
        fragment_energy_twordm *= 0.125

        fragment_energy = fragment_energy_onerdm + fragment_energy_twordm

        return fragment_energy, onerdm

    def _compute_energy_unrestricted(self, dmet_fragment, onerdm, twordm):
        """Calculate the fragment energy (unrestricted mean-field).

        Args:
            dmet_fragment (SecondQuantizedDMETFragment): Self-explanatory.
            onerdm (numpy.array): one-particle reduced density matrix (float).
            twordm (numpy.array): two-particle reduced density matrix (float).

        Returns:
            float64: Fragment energy (fragment_energy).
            numpy.array: One-particle (alpha) RDM for a fragment (onerdm_a, float64).
            numpy.array: One-particle (beta) RDM for a fragment (onerdm_b, float64).
        """

        fock = dmet_fragment.fock
        t_list = dmet_fragment.t_list
        oneint = dmet_fragment.one_ele
        twoint = dmet_fragment.two_ele
        mo_coeff = dmet_fragment.mean_field.mo_coeff

        norb = t_list[0]

        # Calculate the one- and two- RDM for DMET energy calculation (Transform to AO basis)
        onerdm_a, onerdm_b = onerdm

        onerdm_a = mo_coeff[0] @ onerdm_a @ mo_coeff[0].T
        onerdm_b = mo_coeff[1] @ onerdm_b @ mo_coeff[1].T

        twordm_aa, twordm_ab, twordm_bb = twordm

        twordm_aa = np.einsum("pi,ijkl->pjkl", mo_coeff[0], twordm_aa)
        twordm_aa = np.einsum("qj,pjkl->pqkl", mo_coeff[0], twordm_aa)
        twordm_aa = np.einsum("rk,pqkl->pqrl", mo_coeff[0], twordm_aa)
        twordm_aa = np.einsum("sl,pqrl->pqrs", mo_coeff[0], twordm_aa)

        twordm_ab = np.einsum("pi,ijkl->pjkl", mo_coeff[0], twordm_ab)
        twordm_ab = np.einsum("qj,pjkl->pqkl", mo_coeff[0], twordm_ab)
        twordm_ab = np.einsum("rk,pqkl->pqrl", mo_coeff[1], twordm_ab)
        twordm_ab = np.einsum("sl,pqrl->pqrs", mo_coeff[1], twordm_ab)

        twordm_bb = np.einsum("pi,ijkl->pjkl", mo_coeff[1], twordm_bb)
        twordm_bb = np.einsum("qj,pjkl->pqkl", mo_coeff[1], twordm_bb)
        twordm_bb = np.einsum("rk,pqkl->pqrl", mo_coeff[1], twordm_bb)
        twordm_bb = np.einsum("sl,pqrl->pqrs", mo_coeff[1], twordm_bb)

        # Calculate fragment expectation value
        fragment_energy_one = np.einsum("ij,ij->", onerdm_a[: norb, :], fock[: norb, :] + oneint[: norb, :]) \
            + np.einsum("ij,ij->", onerdm_a[:, : norb ], fock[:, : norb] + oneint[:, : norb]) \
            + np.einsum("ij,ij->", onerdm_b[: norb, :], fock[: norb, :] + oneint[: norb, :]) \
            + np.einsum("ij,ij->", onerdm_b[:, : norb ], fock[:, : norb] + oneint[:, : norb])
        fragment_energy_one *= 0.25

        fragment_energy_two = np.einsum("ijkl,ijkl->", twordm_aa[: norb, :, :, :], twoint[: norb, :, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_aa[:, : norb, :, :], twoint[:, : norb, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_aa[:, :, : norb, :], twoint[:, :, : norb, :]) \
            + np.einsum("ijkl,ijkl->", twordm_aa[:, :, :, : norb], twoint[:, :, :, : norb]) \
            + np.einsum("ijkl,ijkl->", twordm_ab[: norb, :, :, :], twoint[: norb, :, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_ab[:, : norb, :, :], twoint[:, : norb, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_ab[:, :, : norb, :], twoint[:, :, : norb, :]) \
            + np.einsum("ijkl,ijkl->", twordm_ab[:, :, :, : norb], twoint[:, :, :, : norb]) \
            + np.einsum("klij,ijkl->", twordm_ab[:, :, : norb, :], twoint[: norb, :, :, :]) \
            + np.einsum("klij,ijkl->", twordm_ab[:, :, :, : norb], twoint[:, : norb, :, :]) \
            + np.einsum("klij,ijkl->", twordm_ab[: norb, :, :, :], twoint[:, :, : norb, :]) \
            + np.einsum("klij,ijkl->", twordm_ab[:, : norb, :, :], twoint[:, :, :, : norb]) \
            + np.einsum("ijkl,ijkl->", twordm_bb[: norb, :, :, :], twoint[: norb, :, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_bb[:, : norb, :, :], twoint[:, : norb, :, :]) \
            + np.einsum("ijkl,ijkl->", twordm_bb[:, :, : norb, :], twoint[:, :, : norb, :]) \
            + np.einsum("ijkl,ijkl->", twordm_bb[:, :, :, : norb], twoint[:, :, :, : norb])
        fragment_energy_two *= 0.125

        fragment_energy = fragment_energy_one + fragment_energy_two

        return fragment_energy, onerdm_a, onerdm_b

    def _default_optimizer(self, func, var_params):
        """Function used as a default optimizer for DMET when user does not
        provide one.

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes var_params as input and returns
                a float.
            var_params (list): The variational parameters (float).

        Returns:
            float: The chemical potential found by the optimizer.
        """

        result = scipy.optimize.newton(func, var_params, tol=1e-5)

        return result.real
