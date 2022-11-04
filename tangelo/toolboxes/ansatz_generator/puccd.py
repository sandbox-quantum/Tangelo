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

"""TBD
V.E. Elfving, M. Millaruelo, J.A. Gámez, and C. Gogolin, Phys. Rev. A 103, 032605 (2021).
"""

import itertools
import numpy as np

from tangelo.linq import Circuit
from tangelo.toolboxes.ansatz_generator import Ansatz
from tangelo.toolboxes.ansatz_generator.ansatz_utils import givens_gate
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit


class pUCCD(Ansatz):
    """TBD.
    """

    def __init__(self, molecule, mapping="JW", up_then_down=False, spin=None, reference_state="HF"):
        pass

    def set_var_params(self, var_params=None):
        """TDB.
        """
        pass

    def prepare_reference_state(self):
        """TBD.
        """
        pass

    def build_circuit(self, var_params=None):
        """TBD.
        """
        pass

    def update_var_params(self, var_params):
        """TBD.
        """
        pass
