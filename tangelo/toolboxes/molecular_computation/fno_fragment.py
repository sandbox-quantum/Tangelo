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

import os
import requests

import h5py
import numpy as np
import pandas as pd


class QEMIST_FNO_fragment():
    """Python object to post-process, fetch and manipulate QEMIST Cloud MI-FNO
    results. The use-case for this is to map MI-FNO subproblems into
    fermionic Hamiltonians acting as inputs. This object keeps track of the
    classical results.

    Attributes:
        e_tot (float): Total MI-FNO energy.
        e_corr (float): Correlation energy (e_tot - e_mf).
        e_mf (float): Mean-field energy.
        frag_info (dict): Information about each fragment. The keys are related
            to the sampled active space (e.g. '(1,)' or '(0, 2)'). It contains
            informations about the correction term, epsilon, list of truncated
            orbitals and more.

    Properties:
        dataframe (pandas.DataFrame): Converted frag_info dict into a pandas
            DataFrame.
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
            "mp2_correlation",
            "correction",
            "frozen_orbitals_truncated",
            "mo_coefficients"
         }

        # Selecting only the FNO keys found in 'relevant_info'.
        self.frag_info = {}
        for _, fragments_per_truncation in result["subproblem_data"].items():
            for frag_id, frag_result in fragments_per_truncation.items():
                self.frag_info[frag_id] = {k: frag_result.get(k, None) for k in relevant_info}

    @property
    def dataframe(self):
        """Outputting the fragment informations as a pandas.DataFrame."""
        return pd.DataFrame.from_dict(self.frag_info, orient="index")

    def get_mo_coeff(self, download_path=os.path.expanduser("~")):
        """Function to fetch molecular orbital coefficients. A download path can
        be provided to change the directory where the files will be downloaded.
        If the files already exist, the function skips the download step. The
        array is stored in the _frag_info[frag_id]["mo_coefficients"]["array"]
        attribute.

        Args:
            download_path (string): Path where to download the HDF5 files
                containing the molecular coefficient array. Default is set to
                the user's HOME directory.
        """
        absolute_path = os.path.abspath(download_path)

        # For each fragment, fetch the molecular orbital coefficients from the
        # HDF5 files.
        for frag_id, frag in self.frag_info.items():
            file_path = os.path.join(absolute_path, frag["mo_coefficients"]["key"] + ".hdf5")

            if not os.path.exists(file_path):
                response = requests.get(frag["mo_coefficients"]["s3_url"])

                with open(file_path, "wb") as file:
                    file.write(response.content)

            with h5py.File(file_path, "r") as file:
                mo_coeff = np.array(file["mo_coefficients"])

            self.frag_info[frag_id]["mo_coefficients"]["array"] = mo_coeff

    def get_fermionoperator(self, molecule, frag_id):

        mo_coeff = self.frag_info[frag_id]["mo_coefficients"]["array"]
        frozen_orbitals = self.frag_info[frag_id]["frozen_orbitals_truncated"]

        # Something is wrong if the molecule provided does not have the same
        # mean-field energy.
        assert round(molecule.mf_energy, 6) == round(self.e_mf, 6)

        # Returning a new molecule with the right frozen orbital.
        new_molecule = molecule.freeze_mos(frozen_orbitals, inplace=False)

        return new_molecule._get_fermionic_hamiltonian(mo_coeff)


if __name__ == "__main__":
    import pickle
    from tangelo import SecondQuantizedMolecule

    with open("/home/alex/scratch/BeH2_MIFNO_trunc1_result.pkl", "rb") as f:
        res = pickle.load(f)

    test = QEMIST_FNO_fragment(res)
    print(test.dataframe)
    print(test.e_tot)
    print(test.e_corr)
    print(test.e_mf)

    test.get_mo_coeff("/home/alex/scratch/mo_coeff")

    BeH2 = f"""
        Be   0.0000    0.0000    0.0000
        H    0.0000    0.0000    1.3264
        H    0.0000    0.0000   -1.3264
    """
    mol = SecondQuantizedMolecule(BeH2, basis="3-21G")
    #print(mol.__dict__)

    ferm_op = test.get_fermionoperator(mol, "(1,)")
    print(ferm_op)
