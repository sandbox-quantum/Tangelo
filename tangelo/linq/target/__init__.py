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

from .backend import Backend
from .target_cirq import CirqSimulator
from .target_qiskit import QiskitSimulator
from .target_qulacs import QulacsSimulator
from .target_qdk import QDKSimulator
from .target_sympy import SympySimulator
from .target_stim import StimSimulator
from tangelo.helpers.utils import all_backends_simulator, clifford_backends_simulator


target_dict = {"qiskit": QiskitSimulator, "cirq": CirqSimulator, "qdk": QDKSimulator, "qulacs": QulacsSimulator, "sympy": SympySimulator, "stim": StimSimulator}

# Generate backend info dictionary
backend_info = {sim_id: target_dict[sim_id].backend_info() for sim_id in all_backends_simulator}
