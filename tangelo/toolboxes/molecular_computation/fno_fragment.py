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

"""Docstring.
"""

import os
import requests

import pandas as pd


class QEMIST_FNO_fragment():

    def __init__(self, result):
        """Docstring."""

        self.e_tot = result["energy_total"]
        self.e_corr = result["energy_correlation"]
        self.e_mf = result["mean_field_energy"]

        frag_info = {
            "n_body": list(),
            "frag_id": list(), "epsilon": list(), "mp2_correlation": list(),
            "correction": list(), "frozen_orbitals_truncated": list()
        }
        for n_body, fragments_per_truncation in result["subproblem_data"].items():
            for frag_id, frag_result in fragments_per_truncation.items():

                frag_info["n_body"] += [n_body]
                frag_info["frag_id"] += [frag_id]
                frag_info["epsilon"] += [frag_result.get("epsilon", 0.)]
                frag_info["mp2_correlation"] += [frag_result.get("mp2_correlation", 0.)]
                frag_info["correction"] += [frag_result.get("correction", 0.)]

                frag_info["frozen_orbitals_truncated"] += [frag_result.get("frozen_orbitals_truncated", None)]
                frag_info["mo_coefficients"] += [frag_result.get("mo_coefficients", None)]

        self.frag_info = frag_info

    @property
    def get_frag_info(self):
        """Docstring."""

        data = {k: v for k, v in self.frag_info.items() if k not in {"frozen_orbitals_truncated"}}
        return pd.DataFrame.from_dict(data, orient="columns")

    def download_mo_coeff(self, download_path=os.getcwd()):
        """Docstring."""

        absolute_path = os.path.abspath(download_path)

        for mo_coeff in self.frag_info["mo_coefficients"]:
            response = requests.get(mo_coeff["s3_url"])
            file_path = os.path.join(absolute_path, mo_coeff["key"])

            with open(file_path, "wb") as file:
                file.write(response.content)


if __name__ == "__main__":
    res = {"energy_correlation":-0.06749160267353815,"energy_total":-15.834543249968815,"mean_field_energy":-15.790965945526066,
    "subproblem_data":{
        "1":{
            "(0,)":{"energy_correlation":-0.00005561322528357948,"energy_total":-15.76710726052056,"epsilon":-0.00005561322528357948, "problem_handle":28268144955910096},
            "(1,)":{"energy_correlation":-0.02942361607204269,"energy_total":-15.79647526336732,"epsilon":-0.02942361607204269, "problem_handle":14851925548623824},
            "(2,)":{"energy_correlation":-0.029423623773858765,"energy_total":-15.796475271069136,"epsilon":-0.029423623773858765}
            },
        "2":{
            "(0, 1)":{"energy_correlation":-0.031521801844350605,"energy_total":-15.798573449139628,"epsilon":-0.002042572547024335,"problem_handle":23162657201612800},
            "(0, 2)":{"energy_correlation":-0.03152180146817507,"energy_total":-15.798573448763452,"epsilon":-0.002042564469032726,"problem_handle":28952788512696320},
            "(1, 2)":{"energy_correlation":-0.06335085243219751,"energy_total":-15.830402499727477,"epsilon":-0.004503612586296057,"problem_handle":20545643433846784}
            }
    }
    }
    test = QEMIST_FNO_fragment(res)
    print(test.get_frag_info)
