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

"""This file provides functions allowing users to import problem decomposition
fragment data from a classical calculation. Those fragments could be inputs to
a quantum computing experiment for quantum chemistry.
"""

from functools import reduce
import itertools
import os
import requests
import warnings

import h5py
import numpy as np
import pandas as pd


class MIFNOFragment():
    """Python object to post-process, fetch and manipulate QEMIST Cloud MI-FNO
    results. The use case for this is to map MI-FNO subproblems into
    fermionic Hamiltonians acting as inputs. This object keeps track of the
    classical results.

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
        dataframe (pandas.DataFrame): Converted frag_info dict into a pandas
            DataFrame.
        fragment_ids (list of string): List of all fragment identifiers.
        frag_info_flattened (dictionary): The nested frag_info without the first
            layer (keys = truncation number).
    """

    def __init__(self, result):
        """Initialization method to process the classical results.

        Args:
            results (dict): Classical computation results (QEMIST Cloud output).
        """

        # IncrementalDecompositon quantities.
        self.e_tot = result["energy_total"]
        self.e_corr = result["energy_correlation"]
        self.e_mf = result["mean_field_energy"]

        relevant_info = {
            "energy_total",
            "energy_correlation",
            "epsilon",
            "correction",
            "frozen_orbitals_truncated",
            "mo_coefficients"
         }

        # Selecting only the FNO keys found in 'relevant_info'.
        self.frag_info = dict()
        for n_body, fragments_per_truncation in result["subproblem_data"].items():
            self.frag_info[n_body] = dict()
            for frag_id, frag_result in fragments_per_truncation.items():
                self.frag_info[n_body][frag_id] = {k: frag_result.get(k, None) for k in relevant_info}

    @property
    def dataframe(self):
        """Outputs the fragment informations as a pandas.DataFrame."""
        df = pd.DataFrame.from_dict(self.frag_info_flattened, orient="index")

        # Replace frozen_orbitals_truncated=None with an empty list.
        df["frozen_orbitals_truncated"] = df["frozen_orbitals_truncated"].apply(lambda d: d if isinstance(d, list) else [])

        return df.drop(["mo_coefficients"], axis=1)

    @property
    def fragment_ids(self):
        """Outputs the fragment ids in a list."""
        return list(itertools.chain.from_iterable([d.keys() for d in self.frag_info.values()]))

    @property
    def frag_info_flattened(self):
        """Outputs the nested frag_info without the first layer."""
        return reduce(lambda a, b: {**a, **b}, self.frag_info.values())

    def retrieve_mo_coeff(self, download_path=os.getcwd()):
        """Function to fetch molecular orbital coefficients. A download path can
        be provided to change the directory where the files will be downloaded.
        If the files already exist, the function skips the download step. The
        array is stored in the ["mo_coefficients"]["array"] entry in the
        frag_info dictionary attribute.

        Args:
            download_path (string): Path where to download the HDF5 files
                containing the molecular coefficient array. Default is set to
                the current work directory.
        """
        absolute_path = os.path.abspath(download_path)

        # For each fragment, fetch the molecular orbital coefficients from the
        # HDF5 files.
        for n_body_fragments in self.frag_info.values():
            for frag_id, frag in n_body_fragments.items():
                file_path = os.path.join(absolute_path, frag["mo_coefficients"]["key"] + ".hdf5")

                if not os.path.exists(file_path):
                    response = requests.get(frag["mo_coefficients"]["s3_url"])

                    with open(file_path, "wb") as file:
                        file.write(response.content)

                with h5py.File(file_path, "r") as file:
                    mo_coeff = np.array(file["mo_coefficients"])

                n_body_fragments[frag_id]["mo_coefficients"]["array"] = mo_coeff

    def compute_fermionoperator(self, molecule, frag_id):
        """Computes the fermionic Hamiltonian for a MI-FNO fragment.

        Args:
            molecule (SecondQuantizedMolecule): Full molecule description.
            frag_id (string): Fragment id, e.g. "(0, )", "(1, 2)", ...
        """

        if not all(["array" in d["mo_coefficients"] for d in self.frag_info_flattened.values()]):
            raise RuntimeError(f"The molecular orbital coefficients are not available. Please call the {self.__class__.__name__}.get_mo_coeff method.")

        n_body = len(eval(frag_id))
        mo_coeff = self.frag_info[n_body][frag_id]["mo_coefficients"]["array"]
        frozen_orbitals = self.frag_info[n_body][frag_id]["frozen_orbitals_truncated"]

        # Something is wrong if the molecule provided does not have the same
        # mean-field energy.
        assert round(molecule.mf_energy, 6) == round(self.e_mf, 6)

        # Returning a new molecule with the right frozen orbital.
        new_molecule = molecule.freeze_mos(frozen_orbitals, inplace=False)

        return new_molecule._get_fermionic_hamiltonian(mo_coeff)

    def mi_summation(self, outside_energies=None):
        """Recomputes the total energy for the method of increments (MI).
        Each increment corresponds to "new" correlation energy from the n-body
        problem. This method makes computing the total energy with new
        results possible.

        Args:
            outside_energies (dict): New energies for a or many fragment(s).
                E.g. {"(0, )": -1.234} or {"(1, )": -1.234, "(0, 1)": -5.678}.
        """
        if outside_energies is None:
            outside_energies = dict()

        fragment_energies = {k: v["energy_total"] for k, v in self.frag_info_flattened.items()}

        # Update to consider energies taken from a calculation.
        fragment_energies.update(outside_energies)

        n_body_max = max(self.frag_info.keys())

        # Computing the epsilon with the MP2 correction.
        # \epsilon_{i} = E_c(i)
        # \epsilon_{ij} = E_c(ij) - \epsilon_{i} - \epsilon_{i}
        # \epsilon_{ijk} = E_c(ijk) - \epsilon_{ij} - \epsilon_{ik}
        #   - \epsilon_{jk} - \epsilon_{i} - \epsilon_{j} - \epsilon_{k}
        # etc.
        epsilons = dict()
        for n_body in range(1, n_body_max + 1):
            for frag_id, result in self.frag_info[n_body].items():
                corr_energy = fragment_energies[frag_id] - self.e_mf
                corr_energy += result["correction"]

                epsilons[frag_id] = corr_energy

                if n_body > 1:
                    for b in range(1, n_body):
                        for c in itertools.combinations(eval(frag_id), b):
                            epsilons[frag_id] -= epsilons[str(c)]

        # Checks if epsilon < 0, i.e. positive correlation energy increment.
        for frag_id, eps in epsilons.items():
            if eps > 0.:
                warnings.warn(f"Epsilon for frag_id {frag_id} is positive ({eps}).", RuntimeWarning)

        return self.e_mf + sum(epsilons.values())
