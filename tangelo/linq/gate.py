# Copyright SandboxAQ 2021-2024.
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

"""Define an abstract quantum gate class holding information about a quantum
gate operation, without tying it to a particular backend or an underlying
mathematical operation.
"""
from math import pi, isclose
from typing import Union

from numpy import integer, ndarray, floating
from sympy import Symbol


ONE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "PHASE"}
TWO_QUBIT_GATES = {"CNOT", "CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPHASE", "XX", "SWAP"}
THREE_QUBIT_GATES = {"CSWAP"}

ONE_TARGET_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "PHASE",
                    "CNOT", "CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPHASE"}
TWO_TARGET_GATES = {"XX", "SWAP", "CSWAP"}

PARAMETERIZED_GATES = {"RX", "RY", "RZ", "PHASE", "CRX", "CRY", "CRZ", "CPHASE", "XX"}
INVERTIBLE_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CH", "PHASE",
                    "CNOT", "CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPHASE", "XX", "SWAP",
                    "CSWAP"}

CLIFFORD_GATES = {"H", "S", "X", "Z", "Y", "SDAG", "CNOT", "CX", "CY", "CZ", "SWAP", "CXSWAP"}


class Gate(dict):
    """An abstract gate class that exposes all the gate information such as gate
    name, control and target qubit indices, parameter values. Assumes qubit
    indexation starts at index 0.

    Attributes:
        name (str): the gate name,
        target (int): A positive integer denoting the index of the target qubit,
        control (int): A positive integer denoting the index of the control
            qubit,
        parameter: A floating-point number or symbolic expression that will
            resolve at runtime. Can hold anything.
        is_variational (bool): a boolean indicating if the gate operation is
            variational or not.
    """

    def __init__(self, name: str, target: Union[int, integer, list, ndarray],
                 control: Union[int, integer, list, ndarray] = None,
                 parameter="", is_variational: bool = False):
        """ This gate class is basically a dictionary with extra methods. """

        def check_qubit_indices(qubit_list, label):
            """Function to check if all given qubit indices are positive integers

            Args:
                qubit_list (list) :: List of values to check are positive integers
                label (str) :: The label of the list, "control" or "target"

            Raises:
                ValueError :: If any of the values in the list are not non-negative integers.
            """
            errmsg = ""
            for ind in qubit_list:
                if (type(ind) != int) or (ind < 0):
                    errmsg += f"\n {label} qubit index {ind} is not a non-negative integer"

            if errmsg:
                raise ValueError(f"Error: type or value of {label} qubit indices not as expected. See details below:"+errmsg)

        target = (target.tolist() if isinstance(target, ndarray) else list(target)) if hasattr(target, "__iter__") else [target]

        check_qubit_indices(target, "target")

        if not isinstance(name, str):
            raise TypeError(f"The name of the gate must be a string but received {type(name)}")
        else:
            name = name.upper()

        if control is not None:
            if name[0] != "C":
                raise ValueError(f"Gate {name} was given control={control} but does not support controls. Try C{name}")
            control = (control.tolist() if isinstance(control, ndarray) else list(control)) if hasattr(control, "__iter__") else [control]
            check_qubit_indices(control, "control")

        # Check for duplications in qubits
        all_involved_qubits = target if control is None else target + control
        if len(all_involved_qubits) != len(set(all_involved_qubits)):
            raise ValueError(f"There are duplicate qubits in the target/control qubits")

        # Check for the number of qubits. If it is a custom gate (not defined in
        # the X_QUBIT_GATES sets), ignore the check. We also do not prevent
        # the creation of multi-controlled gates (begins with "C").
        if name.upper() in ONE_TARGET_GATES:
            n_targets = 1
        elif name.upper() in TWO_TARGET_GATES:
            n_targets = 2
        else:
            n_targets = len(target)
        if len(target) != n_targets:
            raise ValueError(f"Gate {name}: expected {n_targets} target qubits, but got {target}")

        self.__dict__ = {"name": name, "target": target, "control": control,
                         "parameter": parameter, "is_variational": is_variational}

    def __str__(self):
        """Print gate information in a somewhat formatted way. Do not print
        empty attributes.
        """

        mystr = f"{self.name:<10}"
        for attr in ["target", "control"]:
            if self.__getattribute__(attr) or isinstance(self.__getattribute__(attr), int):
                mystr += f"{attr} : {self.__getattribute__(attr)}   "
        if self.__getattribute__("parameter") != "":
            mystr += f"parameter : {self.__getattribute__('parameter')}"
        if self.is_variational:
            mystr += "\t (variational)"

        return mystr

    def __repr__(self):
        """Represent gate such that eval(self.__repr__) returns a proper instance of Gate."""

        mystr = f"Gate(name='{self.name}'"
        for attr in ["target", "control"]:
            if self.__getattribute__(attr) or isinstance(self.__getattribute__(attr), int):
                mystr += f", {attr}={self.__getattribute__(attr)}"
        if self.__getattribute__("parameter") != "":
            parameter = self.__getattribute__('parameter')
            mystr += f", parameter='{parameter}'" if isinstance(parameter, str) else f", parameter={parameter}"
        if self.is_variational:
            mystr += ", is_variational=True"
        mystr += ")"

        return mystr

    def __eq__(self, other):
        """Define equality (==) operator on gates"""

        ds, do = self.__dict__, other.__dict__

        # CNOT and CX gates are equivalent. If both gates are either CNOT or CX, the names are equivalent and can
        # be ignored for equivalence checks later.
        ignore_list = ["name", "parameter"] if ds["name"] in ["CNOT", "CX"] and do["name"] in ["CNOT", "CX"] else ["parameter"]

        if any(ds[k] != do[k] for k in ds if k not in ignore_list):
            return False

        parameter = round(ds["parameter"] % (2 * pi), 7) if isinstance(ds["parameter"], (float, int)) else ds["parameter"]
        other_parameter = round(do["parameter"] % (2 * pi), 7) if isinstance(do["parameter"], (float, int)) else do["parameter"]

        return parameter == other_parameter

    def __ne__(self, other):
        """Define inequality (!=) operator on gates"""
        return not (self == other)

    def inverse(self):
        """Return the inverse (adjoint) of a gate.

        Return:
            Gate: the inverse of the gate.
        """
        if self.name not in INVERTIBLE_GATES:
            raise AttributeError(f"{self.name} is not an invertible gate")

        if self.name in {"T", "S"}:
            new_parameter = -pi / 2 if self.name == "S" else -pi / 4
            return Gate("PHASE", self.target, self.control, new_parameter, self.is_variational)

        if self.parameter == "":
            new_parameter = ""
        elif isinstance(self.parameter, (float, floating, int, integer, Symbol)):
            new_parameter = -self.parameter
        else:
            raise AttributeError(f"{self.name} is not an invertible gate when parameter is {self.parameter}")
        return Gate(self.name, self.target, self.control, new_parameter, self.is_variational)

    def is_clifford(self, abs_tol=1e-4):
        """
        Check if a quantum gate is a Clifford gate.

        Args:
            abs_tol (float) : Optional, Absolute tolerance for the difference between gate parameter clifford parameter values

        Returns:
            bool: True if the gate is in Clifford gate list or has a parameter corresponding to Clifford points,
                  False otherwise.
        """
        if self.name in CLIFFORD_GATES:
            return True
        elif self.name in {"RX", "RY", "RZ", "PHASE"}:
            return isclose(self.parameter % (pi / 2), 0, abs_tol=abs_tol)
        else:
            return False

    def serialize(self):
        return {"type": "Gate",
                "params": {"name": self.name, "target": self.target,
                           "control": self.control, "parameter": self.parameter}}
