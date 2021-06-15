"""
    A circuit translator that can currently take an abstract circuit and build objects allowing to simulate the circuit
    on a target backend. Currently supported: Qiskit, Qulacs, QDK.

    Depending on the API of the target backend, this translator returns a quantum circuit object, a string, or
    even creates a file that will be compiled at runtime by the backend.

    In order to produce an equivalent circuit for the target backend, it is necessary to account for:
    - how the gate names differ between the source backend to the target backend
    - how the order and conventions for some of the inputs to the gate operations may also differ
"""

import re
import qulacs
from qulacs.gate import X, Y, Z, Probabilistic, DepolarizingNoise, TwoQubitDepolarizingNoise
from braket.circuits import Circuit as BraketCircuit
import qiskit
import cirq

from agnostic_simulator import Gate, Circuit

# Dictionary whose keys are the supported backends and the values the gates currently supported by each of them
# TODO: Refactor if needed, figure out how this can best be used by the user and the developer
SUPPORTED_GATES = dict()

# Map gate name of the abstract format to the equivalent add_gate method of Qulacs's QuantumCircuit class
# API and supported gates: http://qulacs.org/class_quantum_circuit.html
GATE_QULACS = dict()
GATE_QULACS["H"] = qulacs.QuantumCircuit.add_H_gate
GATE_QULACS["X"] = qulacs.QuantumCircuit.add_X_gate
GATE_QULACS["Y"] = qulacs.QuantumCircuit.add_Y_gate
GATE_QULACS["Z"] = qulacs.QuantumCircuit.add_Z_gate
GATE_QULACS["S"] = qulacs.QuantumCircuit.add_S_gate
GATE_QULACS["T"] = qulacs.QuantumCircuit.add_T_gate
GATE_QULACS["RX"] = qulacs.QuantumCircuit.add_RX_gate
GATE_QULACS["RY"] = qulacs.QuantumCircuit.add_RY_gate
GATE_QULACS["RZ"] = qulacs.QuantumCircuit.add_RZ_gate
GATE_QULACS["CNOT"] = qulacs.QuantumCircuit.add_CNOT_gate
GATE_QULACS["MEASURE"] = qulacs.gate.Measurement
SUPPORTED_GATES["qulacs"] = sorted(list(GATE_QULACS.keys()))

# Map gate name of the abstract format to the equivalent add_gate method of Qiskit's QuantumCircuit class
# API and supported gates: https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
GATE_QISKIT = dict()
GATE_QISKIT["H"] = qiskit.QuantumCircuit.h
GATE_QISKIT["X"] = qiskit.QuantumCircuit.x
GATE_QISKIT["Y"] = qiskit.QuantumCircuit.y
GATE_QISKIT["Z"] = qiskit.QuantumCircuit.z
GATE_QISKIT["S"] = qiskit.QuantumCircuit.s
GATE_QISKIT["T"] = qiskit.QuantumCircuit.t
GATE_QISKIT["RX"] = qiskit.QuantumCircuit.rx
GATE_QISKIT["RY"] = qiskit.QuantumCircuit.ry
GATE_QISKIT["RZ"] = qiskit.QuantumCircuit.rz
GATE_QISKIT["CNOT"] = qiskit.QuantumCircuit.cx
GATE_QISKIT["MEASURE"] = qiskit.QuantumCircuit.measure
SUPPORTED_GATES["qiskit"] = sorted(list(GATE_QISKIT.keys()))

# Map gate name of the abstract format to the equivalent methods of the braket.circuits.Circuit class
# API and supported gates: https://amazon-braket-sdk-python.readthedocs.io/en/latest/_apidoc/braket.circuits.circuit.html
GATE_BRAKET = dict()
GATE_BRAKET["H"] = BraketCircuit.h
GATE_BRAKET["X"] = BraketCircuit.x
GATE_BRAKET["Y"] = BraketCircuit.y
GATE_BRAKET["Z"] = BraketCircuit.z
GATE_BRAKET["S"] = BraketCircuit.s
GATE_BRAKET["T"] = BraketCircuit.t
GATE_BRAKET["RX"] = BraketCircuit.rx
GATE_BRAKET["RY"] = BraketCircuit.ry
GATE_BRAKET["RZ"] = BraketCircuit.rz
GATE_BRAKET["CNOT"] = BraketCircuit.cnot
#GATE_BRAKET["MEASURE"] = ? (mid-circuit measurement currently unsupported?)
SUPPORTED_GATES["braket"] = sorted(list(GATE_BRAKET.keys()))

# Map gate name of the abstract format to the equivalent methods of the cirq class
# API and supported gates: https://quantumai.google/cirq/gates
GATE_CIRQ= dict()
GATE_CIRQ["H"] = cirq.H
GATE_CIRQ["X"] = cirq.X
GATE_CIRQ["Y"] = cirq.Y
GATE_CIRQ["Z"] = cirq.Z
GATE_CIRQ["S"] = cirq.S
GATE_CIRQ["T"] = cirq.T
GATE_CIRQ["RX"] = cirq.rx 
GATE_CIRQ["RY"] = cirq.ry
GATE_CIRQ["RZ"] = cirq.rz
GATE_CIRQ["CNOT"] = cirq.CNOT
GATE_CIRQ["MEASURE"] = cirq.measure
SUPPORTED_GATES["cirq"] = sorted(list(GATE_CIRQ.keys()))  

# Map gate name of the abstract format to the equivalent gate name used in Q# operations
# API and supported gates: https://projectq.readthedocs.io/en/latest/projectq.ops.html
GATE_PROJECTQ = dict()
for name in {"H", "X", "Y", "Z", "S", "T"}:
    GATE_PROJECTQ[name] = name
for name in {"RX", "RY", "RZ", "MEASURE"}:
    GATE_PROJECTQ[name] = name[0] + name[1:].lower()
GATE_PROJECTQ["CNOT"] = "CX"
SUPPORTED_GATES["projectq"] = sorted(list(GATE_PROJECTQ.keys()))

# Map gate name of the abstract format to the equivalent gate name used in Q# operations
# API and supported gates: https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic
GATE_QDK = dict()
for name in {"H", "X", "Y", "Z", "S", "T", "CNOT"}:
    GATE_QDK[name] = name
for name in {"RX", "RY", "RZ"}:
    GATE_QDK[name] = name[0] + name[1:].lower()
GATE_QDK["MEASURE"] = "M"
SUPPORTED_GATES["qdk"] = sorted(list(GATE_QDK.keys()))

# Map gate name of the abstract format to the equivalent gate name used in openqasm
# OpenQASM is a general format that allows users to express a quantum program, define conditional operations
# manipulating quantum and qubit registers, as well as defining new quantum unitaries.
# We however make the choice here to support well-known gate operations.
GATE_OPENQASM = dict()
for name in {"H", "X", "Y", "Z", "S", "T"}:
    GATE_OPENQASM[name] = name.lower()
for name in {"RX", "RY", "RZ", "MEASURE"}:
    GATE_OPENQASM[name] = name.lower()
GATE_OPENQASM["CNOT"] = "cx"
SUPPORTED_GATES["openqasm"] = sorted(list(GATE_OPENQASM.keys()))

# Map gate name of the abstract format to the equivalent gate name used in the json IonQ format
# For more information:   https://dewdrop.ionq.co/    https://docs.ionq.co
GATE_JSON_IONQ = dict()
for name in {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT"}:
    GATE_JSON_IONQ[name] = name.lower()
SUPPORTED_GATES["json_ionq"] = sorted(list(GATE_JSON_IONQ.keys()))


def translate_qulacs(source_circuit, noise_model=None):
    """ Take in an abstract circuit, return an equivalent qulacs QuantumCircuit instance
        If provided with a noise model, will add noisy gates at translation. Not very useful to look at, as qulacs
        does not provide much information about the noisy gates added when printing the "noisy circuit".

        Args:
            source_circuit: quantum circuit in the abstract format
            noise_model: A NoiseModel object from this package, located in the noisy_simulation subpackage
        Returns:
            qulacs.QuantumCircuit: the corresponding qulacs quantum circuit
    """
    target_circuit = qulacs.QuantumCircuit(source_circuit.width)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.target, -1. * gate.parameter)
        elif gate.name in {"CNOT"}:
            (GATE_QULACS[gate.name])(target_circuit, gate.control, gate.target)
        elif gate.name in {"MEASURE"}:
            gate = (GATE_QULACS[gate.name])(gate.target, gate.target)
            target_circuit.add_gate(gate)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qulacs")

        # Add noisy gates
        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    target_circuit.add_gate(Probabilistic(np, [X(gate.target), Y(gate.target), Z(gate.target)]))
                    if gate.control or gate.control == 0:
                        target_circuit.add_gate(Probabilistic(np, [X(gate.control), Y(gate.control), Z(gate.control)]))
                elif nt == 'depol':
                    if gate.control or gate.control == 0:
                        target_circuit.add_gate(TwoQubitDepolarizingNoise(gate.control, gate.target, (15/16)*np))
                    else:
                        target_circuit.add_gate(DepolarizingNoise(gate.target, (3/4) * np))

    return target_circuit


def translate_qiskit(source_circuit):
    """ Take in an abstract circuit, return an equivalent qiskit QuantumCircuit instance

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            qiskit.QuantumCircuit: the corresponding qiskit quantum circuit
    """
    target_circuit = qiskit.QuantumCircuit(source_circuit.width, source_circuit.width)

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.parameter, gate.target)
        elif gate.name in {"CNOT"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.control, gate.target)
        elif gate.name in {"MEASURE"}:
            (GATE_QISKIT[gate.name])(target_circuit, gate.target, gate.target)
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qiskit")
    return target_circuit


def translate_qsharp(source_circuit, operation="MyQsharpOperation"):
    """ Take in an abstract circuit, generate the corresponding Q# operation (state prep + measurement) string,
        in the appropriate Q# template. The Q# output can be written to file and will be compiled at runtime.

        Args:
            source_circuit: quantum circuit in the abstract format
            operation (str), optional: name of the Q# operation
        Returns:
            The Q# code (operation + template). This needs to be written into a .qs file, and compiled at runtime
    """

    # Prepare Q# operation header
    qsharp_string = ""
    qsharp_string += "@EntryPoint()\n"
    qsharp_string += f"operation {operation}() : Result[] {{\n"
#    qsharp_string += "body (...) {\n\n"
    qsharp_string += f"\tmutable c = new Result[{source_circuit.width}];\n"
    qsharp_string += f"\tusing (qreg = Qubit[{source_circuit.width}]) {{\n"

    # Generate Q# strings with the right syntax, order and values for the gate inputs
    body_str = ""
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}(qreg[{gate.target}]);\n"
        elif gate.name in {"RX", "RY", "RZ"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}({gate.parameter}, qreg[{gate.target}]);\n"
        elif gate.name in {"CNOT"}:
            body_str += f"\t\t{GATE_QDK[gate.name]}(qreg[{gate.control}], qreg[{gate.target}]);\n"
        elif gate.name in {"MEASURE"}:
            body_str += f"\t\tset c w/= {gate.target} <- {GATE_QDK[gate.name]}(qreg[{gate.target}]);\n"
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend qdk")
    qsharp_string += body_str + "\n\t\treturn ForEach(MResetZ, qreg);\n"
    qsharp_string += "\t}\n"
#    qsharp_string += "}\n adjoint auto;\n"
    qsharp_string += "}\n"

    # Fills the template with the Q# operation. The written Q# file will be compiled by the qsharp module at runtime
    from .qdk_template import _qdk_template, _header
    template = _header + _qdk_template.format(operation_name=operation)

    # Return Q# code (template + operation) for simulation. Needs to be written to file and compiled at runtime
    return template + qsharp_string + "\n}\n"


def translate_projectq(source_circuit):
    """ Take in an abstract circuit, return a string containing equivalent projectq instructions

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            projectq_circuit(str): the corresponding projectq instructions (allocation , gates, deallocation)
    """

    projectq_circuit = ""
    for i in range(source_circuit.width):
        projectq_circuit += f"Allocate | Qureg[{i}]\n"

    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T", "MEASURE"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]} | Qureg[{gate.target}]\n"
        elif gate.name in {"RX", "RY", "RZ"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]}({gate.parameter}) | Qureg[{gate.target}]\n"
        elif gate.name in {"CNOT"}:
            projectq_circuit += f"{GATE_PROJECTQ[gate.name]} | ( Qureg[{gate.control}], Qureg[{gate.target}] )\n"
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend projectQ")

    return projectq_circuit


def _translate_projectq2abs(projectq_str):
    """
        Convenience function to help people move away from their ProjectQ workflow.
        Take ProjectQ instructions, expressed as a string, similar to one from the ProjectQ `CommandPrinter` engine.
        Will bundle all qubit allocation (resp. deallocation) at the beginning (resp. end) of the circuit.
        Example available in the `examples` folder.

        Args:
            projectq_str(str): ProjectQ program, as a string (Allocate, Deallocate, gate operations...)
        Returns:
            abs_circ: corresponding quantum circuit in the abstract format
    """

    # Get dictionary of gate mapping, as the reverse dictionary of abs -> projectq translation
    gate_mapping = {v: k for k, v in GATE_PROJECTQ.items()}

    # TODO account for mid-circuit measurements, only ignore final measurements
    # Ignore Measure instructions
    projectq_str = re.sub(r'Measure(.*)\n', '', projectq_str)

    # Ignore allocate and deallocate instructions.
    # Number of qubits is inferred by the abstract circuit, no (de)allocation will occur mid-circuit.
    projectq_str = re.sub(r'(.*)llocate(.*)\n', '', projectq_str)
    projectq_gates = [instruction for instruction in projectq_str.split("\n") if instruction]

    # Translate instructions to abstract gates
    abs_circ = Circuit()
    for projectq_gate in projectq_gates:

        # Extract gate name, qubit indices and parameter value (single parameter for now)
        gate_name = re.split(' \| |\(', projectq_gate)[0]
        qubit_indices = [int(index) for index in re.findall('Qureg\[(\d+)\]', projectq_gate)]
        parameters = [float(index) for index in re.findall('\((.*)\)', projectq_gate) if "Qureg" not in index]

        if gate_name in {"H", "X", "Y", "Z", "S", "T"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[0])
        elif gate_name in {"Rx", "Ry", "Rz"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[0], parameter=parameters[0])
        # #TODO: Rethink the use of enums for gates to set the equality CX=CNOT and enable other refactoring
        elif gate_name in {"CX"}:
            gate = Gate(gate_mapping[gate_name], qubit_indices[1], control=qubit_indices[0])
        else:
            raise ValueError(f"Gate '{gate_name}' not supported with project2abs translation")
        abs_circ.add_gate(gate)

    return abs_circ


def translate_openqasm(source_circuit):
    """ Take in an abstract circuit, return a OpenQASM 2.0 string using IBM Qiskit (they are the reference for OpenQASM)

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            openqasm_string(str): the corresponding OpenQASM program, as per IBM Qiskit
    """
    return translate_qiskit(source_circuit).qasm()


def _translate_openqasm2abs(openqasm_str):
    """ Take an OpenQASM 2.0 string as input (as defined by IBM Qiskit), return the equivalent abstract circuit.
        Only a subset of OpenQASM supported, mostly to be able to go back and forth QASM and abstract representations
        to leverage tools and innovation implemented to work in the QASM format. Not designed to support elaborate
        QASM programs defining their own operations. Compatible with qiskit.QuantumCircuit.from_qasm method.

        Assumes single-qubit measurement instructions only. Final qubit register measurement is implicit.

        Args:
            openqasm_string(str): an OpenQASM program, as a string, as defined by IBM Qiskit
        Returns:
            abs_circ: corresponding quantum circuit in the abstract format
    """
    from math import pi

    def parse_param(s):
        """ Parse parameter as either a float or a string if it's not a float """
        try:
            return float(s)
        except ValueError:
            return s

    # Get number of qubits, extract gate operations
    n_qubits = int(re.findall('qreg q\[(\d+)\];', openqasm_str)[0])
    openqasm_gates = openqasm_str.split(f"qreg q[{n_qubits}];\ncreg c[{n_qubits}];")[-1]
    openqasm_gates = [instruction for instruction in openqasm_gates.split("\n") if instruction]

    # Translate gates
    abs_circ = Circuit()
    for openqasm_gate in openqasm_gates:

        # Extract gate name, qubit indices and parameter value (single parameter for now)
        gate_name = re.split('\s|\(', openqasm_gate)[0]
        qubit_indices = [int(index) for index in re.findall('q\[(\d+)\]', openqasm_gate)]
        parameters = [parse_param(index) for index in re.findall('\((.*)\)', openqasm_gate)]
        # TODO: controlled operation, will need to store value in classical register
        #  bit_indices = [int(index) for index in re.findall('c\[(\d+)\]', openqasm_gate)]

        if gate_name in {"h", "x", "y", "z", "s", "t", "measure"}:
            gate = Gate(gate_name.upper(), qubit_indices[0])
        elif gate_name in {"rx", "ry", "rz"}:
            gate = Gate(gate_name.upper(), qubit_indices[0], parameter=eval(str(parameters[0])))
        # TODO: Rethink the use of enums for gates to set the equality CX=CNOT and enable other refactoring
        elif gate_name in {"cx"}:
            gate = Gate("CNOT", qubit_indices[1], control=qubit_indices[0])
        else:
            raise ValueError(f"Gate '{gate_name}' not supported with openqasm translation")
        abs_circ.add_gate(gate)

    return abs_circ


def translate_json_ionq(source_circuit):
    """ Take in an abstract circuit, return a dictionary following the IonQ JSON format as described below
        https://dewdrop.ionq.co/#json-specification

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            json_ionq_circ (dict): representation of the quantum circuit following the IonQ JSON format
    """

    json_gates = []
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'target': gate.target})
        elif gate.name in {"RX", "RY", "RZ"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'target': gate.target, 'rotation': gate.parameter})
        elif gate.name in {"CNOT"}:
            json_gates.append({'gate': GATE_JSON_IONQ[gate.name], 'target': gate.target, 'control': gate.control})
        else:
            raise ValueError(f"Gate '{gate.name}' not supported with JSON IonQ translation")

    json_ionq_circ = {"qubits": source_circuit.width, 'circuit': json_gates}
    return json_ionq_circ

def translate_braket(source_circuit):
    """ Take in an abstract circuit, return a quantum circuit object as defined in the Python Braket SDK

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            target_circuit (braket.circuits.Circuit): quantum circuit in Python Braket SDK format
    """

    target_circuit = BraketCircuit()

    # Map the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target)
        elif gate.name in {"RX", "RY", "RZ"}:
            (GATE_BRAKET[gate.name])(target_circuit, gate.target, gate.parameter)
        elif gate.name in {"CNOT"}:
            (GATE_BRAKET[gate.name])(target_circuit, control=gate.control, target=gate.target)
        #elif gate.name in {"MEASURE"}:
        # implement if mid-circuit measurement available through Braket later on
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend braket")
    return target_circuit

def translate_cirq(source_circuit, noise_model=None):
    """ Take in an abstract circuit, return an equivalent cirq QuantumCircuit instance

        Args:
            source_circuit: quantum circuit in the abstract format
        Returns:
            target_circuit: a corresponding cirq Circuit. Right now, the 
                            structure is of LineQubit. It is possible in the 
                            future that we may support NamedQubit or GridQubit
    """
    target_circuit = cirq.Circuit()
    #cirq by definition uses labels for qubits, this is one way to automatically generate
    #labels. Could also use GridQubit for square lattice or NamedQubit to name qubits
    qubit_list = cirq.LineQubit.range(source_circuit.width) 
    #Add next line to make sure all qubits are initialized
    #cirq will otherwise only initialize qubits that have gates
    target_circuit.append(cirq.I.on_each(qubit_list))

    # Maps the gate information properly. Different for each backend (order, values)
    for gate in source_circuit._gates:
        if gate.name in {"H", "X", "Y", "Z", "S", "T"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.target]))
        elif gate.name in {"RX", "RY", "RZ"}:
            next_gate=GATE_CIRQ[gate.name](gate.parameter)
            target_circuit.append(next_gate(qubit_list[gate.target]))
        elif gate.name in {"CNOT"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.control],qubit_list[gate.target]))
        elif gate.name in {"MEASURE"}:
            target_circuit.append(GATE_CIRQ[gate.name](qubit_list[gate.target]))
        else:
            raise ValueError(f"Gate '{gate.name}' not supported on backend cirq")

        # Add noisy gates
        if noise_model and (gate.name in noise_model.noisy_gates):
            for nt, np in noise_model._quantum_errors[gate.name]:
                if nt == 'pauli':
                    #Define pauli gate in cirq language
                    depo=cirq.asymmetric_depolarize(np[0],np[1],np[2])
                    target_circuit.append(depo(qubit_list[gate.target]))
                    if gate.control or gate.control == 0:
                        target_circuit.append(depo(qubit_list[gate.control]))
                elif nt == 'depol':
                    if gate.control or gate.control == 0:
                        #define 2-qubit depolarization gate
                        depo=cirq.depolarize(np*15/16,2) #param, num_qubits
                        target_circuit.append(depo(qubit_list[gate.control], qubit_list[gate.target])) #gates targetted
                    else:
                        #define 1-qubit depolarization gate
                        depo=cirq.depolarize(np*3/4,1) 
                        target_circuit.append(depo(qubit_list[gate.target]))
                    
    return target_circuit