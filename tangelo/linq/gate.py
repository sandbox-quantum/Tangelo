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

"""Define an abstract quantum gate class holding information about a quantum
gate operation, without tying it to a particular backend or an underlying
mathematical operation.
"""

from typing import Union

from numpy import integer, ndarray

ONE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "PHASE"}
TWO_QUBIT_GATES = {"CNOT", "CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPHASE", "XX", "SWAP"}
THREE_QUBIT_GATES = {"CSWAP"}


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

    def __init__(self, name: str, target: Union[int, integer, list, ndarray], control: Union[int, integer, list, ndarray] = None,
                 parameter="", is_variational: bool = False):
        """ This gate class is basically a dictionary with extra methods. """

        if not isinstance(target, (int, integer, list, ndarray)):
            raise ValueError("Qubit index must be int or list of ints.")
        else:
            if isinstance(target, (int, integer)):
                target = [int(target)]
            new_target = []
            for t in target:
                if isinstance(t, integer) and t >= 0:
                    new_target.append(int(t))
                elif isinstance(t, int) and t >= 0:
                    new_target.append(t)
                else:
                    raise ValueError(f"Target {t} of requested target={target} is not a positive integer")
            target = new_target

        if control is not None:
            if name[0] != "C":
                raise ValueError(f"Gate {name} was given control={control} but does not support controls. Try C{name}")
            if not isinstance(control, (int, integer, list, ndarray)):
                raise ValueError("Qubit index must be int or list of ints.")
            else:
                if isinstance(control, (int, integer)):
                    control = [int(control)]
                new_control = []
                for c in control:
                    if isinstance(c, integer) and c >= 0:
                        new_control.append(int(c))
                    elif isinstance(c, int) and c >= 0:
                        new_control.append(c)
                    else:
                        raise ValueError(f"Control {c} of requested control={control} is not a positive integer")
                control = new_control

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

    def serialize(self):
        return {"type": "Gate",
                "params": {"name": self.name, "target": self.target,
                           "control": self.control, "parameter": self.parameter}}
