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
"""


from tangelo.problem_decomposition.incremental.incremental_helper import MethodOfIncrementsHelper


class MIFNOHelper(MethodOfIncrementsHelper):
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
            mifno_full_result (dict): MIFNO results (QEMIST Cloud output).
        """

        super().__init__(log_file=mifno_log_file, full_result=mifno_full_result)

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
        self.frag_info = MethodOfIncrementsHelper.read_relevant_info(self.full_result, fragment_relevant_info)

        # Performing some checks here.
