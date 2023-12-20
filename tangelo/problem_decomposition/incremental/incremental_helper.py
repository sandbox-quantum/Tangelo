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

"""This file provides helpers in order to import data coming from an incremental
job (MIFNO or iFCI) run on QEMIST Cloud, providing the users with both fragment
information as well as reference results obtained by the classical solvers in
QEMIST Cloud. The fragments can be passed to a quantum solver or be used for a
quantum computing experiment.

MIFNO references:
- Verma, L. Huntington, M. P. Coons, Y. Kawashima, T. Yamazaki and A. Zaribafiyan
The Journal of Chemical Physics 155, 034110 (2021).

iFCI references:
- Paul M. Zimmerman, J. Chem. Phys., 146, 104102 (2017).
- Paul M. Zimmerman, J Chem. Phys., 146, 224104 (2017).
- Alan E. Rask and Paul M. Zimmerman, J. Phys. Chem. A, 125, 7, 1598-1609 (2021).
"""

from functools import reduce
import itertools
import os
import json
import warnings

import h5py
import numpy as np
import pandas as pd


class MethodOfIncrementsHelper():
    """Python class to post-process, fetch and manipulate QEMIST Cloud
    incremental results. This is referring to the Method of Increments (MI).
    MI-FNO and iFCI results are both supported by this class.

    Attributes:
        e_tot (float): Total incremental energy.
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
        fragment_ids (list of str): List of all fragment identifiers.
        frag_info_flattened (dictionary): The nested frag_info without the first
            layer (keys = truncation number).
    """

    def __init__(self, log_file=None, full_result=None):
        """Initialization method to process the classical results. A json path
        or a python dictionary can be passed to the method for the full MI
        results. Passing both a path and a dictionary raises an error. Not all
        fragment results need to be imported: in case of missing data, this
         helper class raises an error mentionning the missing pieces.

        Args:
            log_file (str): Path to a json file containing the MI results
                from QEMIST Cloud.
            full_result (dict): MI results (QEMIST Cloud output).
        """

        # Raise error if input is not as expected. Only a single input must be
        # provided to avoid conflicts.
        if not (bool(log_file) ^ bool(full_result)):
            raise ValueError(f"A file path to the log file OR the full result \
                               dictionary object must be provided when \
                               instantiating {self.__class__.__name__}.")

        if log_file:
            assert os.path.isfile(log_file), f"The file {log_file} does not exist."

            with open(log_file, "r") as f:
                full_result = json.loads("\n".join(f.readlines()[1:]))

        full_result["subproblem_data"] = {int(k): v for k, v in full_result["subproblem_data"].items()}

        # Incremental (problem decomposition) quantities.
        self.e_tot = full_result["energy_total"]
        self.e_corr = full_result["energy_correlation"]
        self.e_mf = self.e_tot - self.e_corr

        self.frag_info = MethodOfIncrementsHelper.read_mi_info(full_result)

    @staticmethod
    def read_mi_info(full_result):
        """Method to filter the relevant information in the full_result
        dictionary.

        Args:
            full_result (dict): QEMIST Cloud output for the MI problem.

        Returns:
            (dict): Simplified version of the MI result dictionary.
        """

        # Relevant info and their default fallback values.
        fragment_relevant_info = {
            "energy_total": 0.,
            "energy_correlation": 0.,
            "frozen_orbitals_truncated": list(),
            "complete_orbital_space": list(),
            "epsilon": 0.,
            "correction": 0.,
            "problem_handle": None,
        }

        # Select only the relevant information in the full result.
        frag_info = dict()
        for n_body, fragments_per_truncation in full_result["subproblem_data"].items():
            frag_info[n_body] = dict()

            for frag_id, frag_result in fragments_per_truncation.items():

                # There is no problem_handle for the fragment if it has been
                # screened out by QEMIST Cloud.
                if frag_result.get("problem_handle", None) is not None:

                    # Default values.
                    frag_info[n_body][frag_id] = fragment_relevant_info.copy()

                    # Updating default values with QEMIST Cloud results.
                    frag_info[n_body][frag_id].update({k: v for k, v in frag_result.items() if k in fragment_relevant_info.keys()})

        return frag_info

    def __repr__(self):
        """Format the object to print the energies and the fragment information
        as a pandas.DataFrame.
        """

        data_fragments = self.to_dataframe
        data_fragments.drop(["problem_handle"], axis=1, inplace=True)
        data_fragments.drop(["frozen_orbitals_truncated"], axis=1, inplace=True)
        data_fragments.drop(["complete_orbital_space"], axis=1, inplace=True)
        str_rep = f"(All the energy values are in hartree)\n" \
                  f"Total incremental energy = {self.e_tot}\n" \
                  f"Correlation energy = {self.e_corr}\n" \
                  f"Mean-field energy = {self.e_mf}\n" \
                  f"{data_fragments}"

        return str_rep

    def __getitem__(self, frag_id):
        """The user can select the fragment information (python dictionary) with
        the [] operator.

        Args:
            frag_id (str): Fragment id, e.g. "(1,)", "(1 ,2)", etc.

        Returns:
            (dict): Fragment information for the provided frag_id.

        """
        return self.frag_info_flattened[frag_id]

    @property
    def to_dataframe(self):
        """Outputs fragment information as a pandas.DataFrame."""
        df = pd.DataFrame.from_dict(self.frag_info_flattened, orient="index")

        # Replace frozen_orbitals_truncated=None with an empty list.
        df["frozen_orbitals_truncated"] = df["frozen_orbitals_truncated"].apply(lambda d: d if isinstance(d, list) else [])
        df["complete_orbital_space"] = df["complete_orbital_space"].apply(lambda d: d if isinstance(d, list) else [])

        df["n_electrons"], df["n_spinorbitals"] = zip(*df.index.map(self.n_electrons_spinorbs))

        return df

    @property
    def fragment_ids(self):
        """Output the fragment ids in a list."""
        return list(itertools.chain.from_iterable([d.keys() for d in self.frag_info.values()]))

    @property
    def frag_info_flattened(self):
        """Output the nested frag_info without the first layer."""
        return reduce(lambda a, b: {**a, **b}, self.frag_info.values())

    def retrieve_mo_coeff(self, source_folder=os.getcwd(), prefix="mo_coefficients_", suffix=".h5"):
        """Function to fetch molecular orbital coefficients. The array is
        stored in the ["mo_coefficients"] entry in the frag_info dictionary
        attribute. Each MO coefficient file name should contain the QEMIST Cloud
        problem handle for this fragment.

        Args:
            source_folder (str): Users can specify a path to a folder, where the
                files containing the coefficients are stored. The default value
                is the directory where the user's python script is run.
            prefix (str): Prefix for the file names. Default is
                "mo_coeffcients_".
            suffix (str):  Suffix for the file name structure, including the
                file extension. Default is ".h5".
        """

        absolute_path = os.path.abspath(source_folder)
        if not os.path.isdir(absolute_path):
            raise FileNotFoundError(f"Folder not found:\n {absolute_path}")

        # For each fragment, fetch the molecular orbital coefficients from the
        # HDF5 files.
        for n_body_fragments in self.frag_info.values():
            for frag_id, frag in n_body_fragments.items():
                file_path = os.path.join(absolute_path, prefix + str(frag["problem_handle"]) + suffix)

                # Files must be downloaded a priori.
                if not os.path.exists(file_path):

                    # This is not important if the user does not request a
                    # fermionic operator for this fragment. In the other case,
                    # an error will be raise in the compute_fermionoperator
                    # method.
                    warnings.warn(f"File {file_path} not found. MO coefficients for fragment {frag_id} are not available.")
                    continue

                with h5py.File(file_path, "r") as file:
                    mo_coeff = np.array(file["mo_coefficients"])

                n_body_fragments[frag_id]["mo_coefficients"] = mo_coeff

    def compute_fermionoperator(self, molecule, frag_id):
        """Compute the fermionic Hamiltonian for an incremental fragment.

        Args:
            molecule (SecondQuantizedMolecule): Full molecule description.
            frag_id (str): Fragment id, e.g. "(0, )", "(1, 2)", ...

        Returns:
            FermionOperator: Fermionic operator for the specified fragment id.
        """

        n_body = len(eval(frag_id))

        if "mo_coefficients" not in self.frag_info[n_body][frag_id]:
            raise RuntimeError(f"The molecular orbital coefficients are not available. "
                               f"Please call the {self.__class__.__name__}.retrieve_mo_coeff method.")

        mo_coeff = self.frag_info[n_body][frag_id]["mo_coefficients"]
        frozen_orbitals = self.frag_info[n_body][frag_id]["frozen_orbitals_truncated"]

        # Returning a new molecule with the frozen orbitals.
        try:
            new_molecule = molecule.freeze_mos(frozen_orbitals, inplace=False)
        except ValueError:
            raise ValueError(f"All orbitals except {frag_id} are frozen from "
                             "the virtual orbital truncation. That means no "
                             "correlation energy can be extracted from this "
                             "fragment.")

        return new_molecule._get_fermionic_hamiltonian(mo_coeff)

    def mi_summation(self, user_provided_energies=None):
        r"""Recompute the total energy for the method of increments (MI).
        Each increment corresponds to "new" correlation energy from the n-body
        problem. This method makes computing the total energy with new
        results possible.

        It computes the epsilons:
        \epsilon_{i} = E_c(i)
        \epsilon_{ij} = E_c(ij) - \epsilon_{i} - \epsilon_{i}
        \epsilon_{ijk} = E_c(ijk) - \epsilon_{ij} - \epsilon_{ik}
            - \epsilon_{jk} - \epsilon_{i} - \epsilon_{j} - \epsilon_{k}
        etc.

        A correction term per fragment is considered if applicable. For MI-FNO,
        there is an MP2 correction for the truncated virtual space. For iFCI,
        there is no such correction.

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
                "recompute the total incremental energy.")

        if user_provided_energies is None:
            user_provided_energies = dict()
        else:
            fragment_correction = {k: v["correction"] for k, v in self.frag_info_flattened.items()}
            user_provided_energies = {frag_id: e + fragment_correction[frag_id] for frag_id, e in user_provided_energies.items()}

        # Update to consider energies taken from a calculation.
        fragment_energies.update(user_provided_energies)

        # Equivalent to truncation_order in QEMIST Cloud.
        n_body_max = max(self.frag_info.keys())

        # Perform the incremental summation.
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
            frag_id (str): Fragment id, e.g. "(0, )", "(1, 2)", ...

        Returns:
            int, int: Number of electrons, number of spinorbitals.
        """

        fragment_info = self.frag_info_flattened[frag_id]

        n_electrons = 2 * len(eval(frag_id))
        n_spinorbs = 2 * (len(fragment_info["complete_orbital_space"]) - len(fragment_info["frozen_orbitals_truncated"]))

        return n_electrons, n_spinorbs
