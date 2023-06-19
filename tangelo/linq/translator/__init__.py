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

from tangelo.helpers.utils import installed_backends

from .translate_braket import get_braket_gates
from .translate_qiskit import get_qiskit_gates
from .translate_qulacs import get_qulacs_gates
from .translate_cirq import get_cirq_gates
from .translate_json_ionq import get_ionq_gates
from .translate_qdk import get_qdk_gates
from .translate_projectq import get_projectq_gates
from .translate_openqasm import get_openqasm_gates
from .translate_pennylane import get_pennylane_gates
from .translate_sympy import get_sympy_gates
from .translate_qubitop import translate_operator
from .translate_circuit import translate_circuit


def get_supported_gates():
    """List all supported gates for all backends found"""

    supported_gates = dict()
    supported_gates["projectq"] = sorted(get_projectq_gates().keys())
    supported_gates["ionq"] = sorted(get_ionq_gates().keys())
    supported_gates["qdk"] = sorted(get_qdk_gates().keys())
    supported_gates["openqasm"] = sorted(get_openqasm_gates().keys())
    supported_gates["qulacs"] = sorted(get_qulacs_gates().keys())
    supported_gates["qiskit"] = sorted(get_qiskit_gates().keys())
    supported_gates["cirq"] = sorted(get_cirq_gates().keys())
    supported_gates["braket"] = sorted(get_braket_gates().keys())
    supported_gates["pennylane"] = sorted(get_pennylane_gates().keys())
    supported_gates["sympy"] = sorted(get_sympy_gates().keys())

    return supported_gates
