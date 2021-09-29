from qsdk.helpers.utils import installed_backends

from .translate_braket import translate_braket, get_braket_gates
from .translate_qiskit import translate_qiskit, get_qiskit_gates
from .translate_qulacs import translate_qulacs, get_qulacs_gates
from .translate_cirq import translate_cirq, get_cirq_gates
from .translate_json_ionq import translate_json_ionq, get_ionq_gates
from .translate_qdk import translate_qsharp, get_qdk_gates
from .translate_projectq import translate_projectq, _translate_projectq2abs, get_projectq_gates
from .translate_openqasm import translate_openqasm, _translate_openqasm2abs, get_openqasm_gates


# List all supported gates for all backends found
SUPPORTED_GATES = dict()

SUPPORTED_GATES["projectq"] = sorted(get_projectq_gates().keys())
SUPPORTED_GATES["ionq"] = sorted(get_ionq_gates().keys())
SUPPORTED_GATES["qdk"] = sorted(get_qdk_gates().keys())
SUPPORTED_GATES["openqasm"] = sorted(get_openqasm_gates().keys())

for v in installed_backends:
    SUPPORTED_GATES[v] = sorted(eval(f'get_{v}_gates().keys()'))
