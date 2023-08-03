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

"""This file provides helpers in order to import data coming from a MI-FNO job
run on QEMIST Cloud, providing the users with both fragment information as well as
reference results obtained by the classical solvers in QEMIST Cloud. The
fragments can be passed to a quantum solver or be used for a quantum computing
experiment.

Currently, the fragment energies can only be recomputed with a quantum
algorihtms (the interface between MI-FNO fragments and classical algorithms is not
implemented yet).
"""

from functools import reduce
import itertools
import os
import requests
import warnings
import json

import h5py
import numpy as np
import pandas as pd


class MIFNOHelper():
    """Python object to post-process, fetch and manipulate QEMIST Cloud MI-FNO
    results. The use case for this is to map MI-FNO subproblems to
    fermionic Hamiltonians acting as inputs. This object keeps track of the
    classical calculation results.

    Attributes:
        e_tot (float): Total MI-FNO energy.
        e_corr (float): Correlation energy (e_tot - e_mf).
        e_mf (float): Mean-field energy.
        frag_info (dict): Information about each fragment. The keys are related
            to the truncation number (int) . The nested dictionaries have keys
             refering to the sampled active space (e.g. '(1,)' or '(0, 2)') They
            contain information about the correction term, epsilon, list of
            truncated orbitals and more.

    Properties:
        to_dataframe (pandas.DataFrame): Converted frag_info dict into a pandas
            DataFrame.
        fragment_ids (list of string): List of all fragment identifiers.
        frag_info_flattened (dictionary): The nested frag_info without the first
            layer (keys = truncation number).
    """

    def __init__(self, mifno_log_file=None, mifno_full_result=None):
        """Initialization method to process the classical results. A json path
        or a python dictionary can be passed to the method for the MI
        or each FNO fragment results. Passing both a path and a dictionary
        raises an error. Not all fragment results need to be imported:
        in case of missing data, MIFNOHelper raises an error
        mentionning the missing pieces.

        Args:
            mifno_log_file (string): Path to a json file containing the MIFNO
                results from QEMIST Cloud.
            mifno_full_result (dict): MI results (QEMIST Cloud output).
        """

        # Raise errorif input is not as expected. Only a single input must be
        # provided to avoid conflicts.
        if not (bool(mifno_log_file) ^ bool(mifno_full_result)):
            raise ValueError(f"A file path to the log file OR the full result \
                               dictionary object for MIFNO must be provided when \
                               instantiating {self.__class__.__name__}.")

        if mifno_log_file:
            assert os.path.isfile(mifno_log_file), f"The file {mifno_log_file} does not exist."

            with open(mifno_log_file, "r") as f:
                mifno_full_result = json.loads(f.read())

        mifno_full_result["subproblem_data"] = {int(k): v for k, v in mifno_full_result["subproblem_data"].items()}

        # Incremental (problem decomposition) quantities.
        self.e_tot = mifno_full_result["energy_total"]
        self.e_corr = mifno_full_result["energy_correlation"]
        self.e_mf = self.e_tot - self.e_corr

        fragment_relevant_info = {
            "energy_total",
            "energy_correlation",
            "correction",
            "frozen_orbitals_truncated",
            "complete_orbital_space",
            "mo_coefficients",
            "epsilon",
            "problem_handle"
        }

        # Select only the relevant information in the full result.
        self.frag_info = dict()
        for n_body, fragments_per_truncation in mifno_full_result["subproblem_data"].items():
            self.frag_info[n_body] = dict()
            for frag_id, frag_result in fragments_per_truncation.items():
                if frag_result.get("problem_handle", None) is not None:
                    self.frag_info[n_body][frag_id] = {k: frag_result.get(k, None) for k in fragment_relevant_info}

                    # If the mean-field energy for the fragment is not the same,
                    # the coordinates, spin, basis or charge of the problem
                    # might be different.
                    e_mf_frag = frag_result["energy_total"] - frag_result["energy_correlation"]
                    assert round(self.e_mf, 6) == round(e_mf_frag, 6), \
                        f"The mean-field energy for fragment {frag_id} is " \
                        "different than the one detected in the MI results."

                    # The mo_coefficients data are essential to recompute
                    # fermionic operators.
                    assert "mo_coefficients" in frag_result, "MO coefficients "\
                        f"not found in the {frag_id} results. Verify that " \
                        "the `export_fragment_data` flag is set to True for " \
                        "the MI-FNO calculation in QEMIST Cloud."

    def __repr__(self):
        """Format the object to print the energies and the fragment information
        as a pandas.DataFrame.
        """

        data_fragments = self.to_dataframe
        data_fragments.drop(["problem_handle"], axis=1, inplace=True)
        data_fragments.drop(["frozen_orbitals_truncated"], axis=1, inplace=True)
        data_fragments.drop(["complete_orbital_space"], axis=1, inplace=True)
        str_rep = f"(All the energy values are in hartree)\n" \
                  f"Total MI-FNO energy = {self.e_tot}\n" \
                  f"Correlation energy = {self.e_corr}\n" \
                  f"Mean-field energy = {self.e_mf}\n" \
                  f"{data_fragments}"

        return str_rep

    def __getitem__(self, frag_id):
        """The user can select the fragment information (python dictionary) with
        the [] operator.
        """
        return self.frag_info_flattened[frag_id]

    @property
    def to_dataframe(self):
        """Outputs fragment information as a pandas.DataFrame."""
        df = pd.DataFrame.from_dict(self.frag_info_flattened, orient="index")

        # Replace frozen_orbitals_truncated=None with an empty list.
        df["frozen_orbitals_truncated"] = df["frozen_orbitals_truncated"].apply(lambda d: d if isinstance(d, list) else [])
        df["complete_orbital_space"] = df["complete_orbital_space"].apply(lambda d: d if isinstance(d, list) else [])

        return df.drop(["mo_coefficients"], axis=1)

    @property
    def fragment_ids(self):
        """Output the fragment ids in a list."""
        return list(itertools.chain.from_iterable([d.keys() for d in self.frag_info.values()]))

    @property
    def frag_info_flattened(self):
        """Output the nested frag_info without the first layer."""
        return reduce(lambda a, b: {**a, **b}, self.frag_info.values())

    def retrieve_mo_coeff(self, destination_folder=os.getcwd()):
        """Function to fetch molecular orbital coefficients. A download path can
        be provided to change the directory where the files will be downloaded.
        If the files already exist, the function skips the download step. The
        array is stored in the ["mo_coefficients"]["array"] entry in the
        frag_info dictionary attribute.

        Args:
            destination_folder (string): Users can specify a path to a
                destination folder, where the files containing the coefficients
                will be downloaded. The default value is the directory where the
                user's python script is run.
        """
        if not os.path.isdir(destination_folder):
            raise FileNotFoundError(f"The {destination_folder} path does not exist.")
        absolute_path = os.path.abspath(destination_folder)

        # For each fragment, fetch the molecular orbital coefficients from the
        # HDF5 files.
        n_files = len(self.fragment_ids)
        i_file = 1
        for n_body_fragments in self.frag_info.values():
            for frag_id, frag in n_body_fragments.items():
                if frag.get("mo_coefficients", None):
                    file_path = os.path.join(absolute_path, str(frag["problem_handle"]) + ".h5")

                    if not os.path.exists(file_path):
                        print(f"Downloading and writing MO coefficients file to {file_path} ({i_file} / {n_files})")
                        response = requests.get(frag["mo_coefficients"]["url"])

                        with open(file_path, "wb") as file:
                            file.write(response.content)

                    with h5py.File(file_path, "r") as file:
                        mo_coeff = np.array(file["mo_coefficients"])

                    n_body_fragments[frag_id]["mo_coefficients"]["array"] = mo_coeff
                else:
                    print(f"MO coefficients for fragment {frag_id} ({i_file} / {n_files}) not available.")

                i_file += 1

    def compute_fermionoperator(self, molecule, frag_id):
        """Compute the fermionic Hamiltonian for a MI-FNO fragment.

        Args:
            molecule (SecondQuantizedMolecule): Full molecule description.
            frag_id (string): Fragment id, e.g. "(0, )", "(1, 2)", ...

        Returns:
            FermionOperator: Fermionic operator for the specified fragment id.
        """
        n_body = len(eval(frag_id))

        if self.frag_info[n_body][frag_id]["mo_coefficients"] is None:
            raise RuntimeError(f"The fragment information has not been imported.")

        if self.frag_info[n_body][frag_id]["mo_coefficients"].get("array", None) is None:
            raise RuntimeError(f"The molecular orbital coefficients are not available. Please call the {self.__class__.__name__}.retrieve_mo_coeff method.")

        mo_coeff = self.frag_info[n_body][frag_id]["mo_coefficients"]["array"]
        frozen_orbitals = self.frag_info[n_body][frag_id]["frozen_orbitals_truncated"]

        # Something is wrong if the molecule provided does not have the same
        # mean-field energy.
        assert round(molecule.mf_energy, 6) == round(self.e_mf, 6),  \
            "The molecule's mean-field energy is different than the one from " \
            "the results. Please verify that the molecular quantities are "\
            "the same as the one in the MI-FNO computation."

        # Returning a new molecule with the frozen orbitals.
        try:
            new_molecule = molecule.freeze_mos(frozen_orbitals, inplace=False)
        except ValueError:
            raise ValueError(f"All orbitals except {frag_id} are frozen from "
                             "the FNO truncation. That means no "
                             "correlation energy can be extracted from this "
                             "fragment.")

        return new_molecule._get_fermionic_hamiltonian(mo_coeff)

    def mi_summation(self, user_provided_energies=None):
        r"""Recompute the total energy for the method of increments (MI).
        Each increment corresponds to "new" correlation energy from the n-body
        problem. This method makes computing the total energy with new
        results possible.

        It computes the epsilons with the MP2 correction:
        \epsilon_{i} = E_c(i)
        \epsilon_{ij} = E_c(ij) - \epsilon_{i} - \epsilon_{i}
        \epsilon_{ijk} = E_c(ijk) - \epsilon_{ij} - \epsilon_{ik}
            - \epsilon_{jk} - \epsilon_{i} - \epsilon_{j} - \epsilon_{k}
        etc.

        Args:
            user_provided_energies (dict): New energy values provided by the
                user, used instead of the corresponding pre-computed ones. E.g.
                {"(0, )": -1.234} or {"(1, )": -1.234, "(0, 1)": -5.678}.

        Returns:
            float: Method of increment total energy.
        """
        fragment_energies = {k: v["energy_total"] for k, v in self.frag_info_flattened.items()}

        if any([e is None for e in fragment_energies.values()]):
            raise ValueError("All fragment data must be imported to "
                "recompute the total MI-FNO energy.")

        if user_provided_energies is None:
            user_provided_energies = dict()
        else:
            fragment_correction = {k: v["correction"] for k, v in self.frag_info_flattened.items()}

            if any(fragment_correction[frag_id] is None for frag_id in user_provided_energies.keys()):
                raise RuntimeError(f"Not all the fragments in {list(user_provided_energies.keys())} "
                    "have been imported. The MP2 correction must be known "
                    "for all fragments to recompute the total MI-FNO energy.")

            user_provided_energies = {frag_id: e + fragment_correction[frag_id] for frag_id, e in user_provided_energies.items()}

        # Update to consider energies taken from a calculation.
        fragment_energies.update(user_provided_energies)

        # Equivalent to truncation_order in QEMIST Cloud.
        n_body_max = max(self.frag_info.keys())

        # Perform the incremental sumamtion.
        epsilons = dict()
        for n_body in range(1, n_body_max + 1):
            for frag_id in self.frag_info[n_body].keys():
                corr_energy = fragment_energies[frag_id] - self.e_mf
                epsilons[frag_id] = corr_energy

                if n_body > 1:
                    for n_increment in range(1, n_body):
                        for frag_increment in itertools.combinations(eval(frag_id), n_increment):
                            epsilons[frag_id] -= epsilons[str(frag_increment)]

        return self.e_mf + sum(epsilons.values())

    def n_electrons_spinorbs(self, frag_id):
        """Output the number of electrons and spinorbitals for a given fragment.

        Args:
            frag_id (string): Fragment id, e.g. "(0, )", "(1, 2)", ...

        Returns:
            int, int: Number of electrons, number of spinorbitals.
        """

        fragment_info = self.frag_info_flattened[frag_id]

        n_electrons = 2 * len(eval(frag_id))
        n_spinorbs = 2 * (len(fragment_info["complete_orbital_space"]) - len(fragment_info["frozen_orbitals_truncated"]))

        return n_electrons, n_spinorbs
