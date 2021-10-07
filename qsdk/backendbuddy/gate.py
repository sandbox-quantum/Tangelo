# Copyright 2021 1QB Information Technologies Inc.
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

from typing import Union

ONE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"}
TWO_QUBIT_GATES = {"CNOT"}


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

    # TODO: extend control to a list to support gates such as the Toffoli gate etc in the future
    # TODO: extend target to a list to support gates such as U2, U3 etc in the future
    def __init__(self, name: str, target: Union[int, list], control: int = None, parameter="", is_variational: bool=False):
        """ This gate class is basically a dictionary with extra methods. """

        if not isinstance(target, (int, list)):
            raise ValueError("Qubit index must be int or list of ints.")
        else:
            if isinstance(target, int):
                if target < 0:
                    raise ValueError("Qubit index must be a positive integer")
                target0 = target
                target1 = None
            elif isinstance(target, list) and len(target) == 2:
                if target[0] >= 0 and target[1] >= 0:
                    target0 = target[0]
                    target1 = target[1]
                else:
                    raise ValueError("Qubit indices must both be positive integers")
            else:
                raise ValueError("Only two target indices are supported")
        if control and (not (isinstance(control, int) and control >= 0)):
            raise ValueError("Qubit index must be a positive integer.")

        self.__dict__ = {"name": name, "target": target0, "target1": target1, "control": control,
                         "parameter": parameter, "is_variational": is_variational}

    def __str__(self):
        """Print gate information in a somewhat formatted way. Do not print
        empty attributes.
        """

        mystr = f"{self.name:<10}"
        for attr in ["target", "target1", "control"]:
            if self.__getattribute__(attr) or isinstance(self.__getattribute__(attr), int):
                mystr += f"{attr} : {self.__getattribute__(attr)}   "
        if self.__getattribute__("parameter") != "":
            mystr += f"parameter : {self.__getattribute__('parameter')}"
        if self.is_variational:
            mystr += "\t (variational)"

        return mystr

    def serialize(self):
        return {"type": "Gate",
                "params": {"name": self.name, "target": self.target, "target1": self.target1,
                           "control": self.control, "parameter": self.parameter}}
