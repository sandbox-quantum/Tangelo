"""
    Define an abstract quantum gate class holding information about a quantum gate operation, without tying it
    to a particular backend or an underlying mathematical operation.
"""

ONE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"}
TWO_QUBIT_GATES = {"CNOT"}


class Gate(dict):
    """
    An abstract gate class that exposes all the gate information such as gate name, control and target qubit indices,
    parameter values. Assumes qubit indexation starts at index 0.

    Attributes:
        name (str): the gate name
        target (int): A positive integer denoting the index of the target qubit
        control (int): A positive integer denoting the index of the control qubit
        parameter: A floatting-point number or symbolic expression that will resolve at runtime. Can hold anything.
        is_variational (bool): a boolean indicating if the gate operation is variational or not
    """

    # TODO: extend control to a list to support gates such as the Toffoli gate etc in the future
    # TODO: extend target to a list to support gates such as U2, U3 etc in the future
    def __init__(self, name: str, target: int, control: int=None, parameter="", is_variational: bool=False):
        """ This gate class is basically a dictionary with extra methods. """

        if not (isinstance(target, int) and target >= 0):
            raise ValueError("Qubit index must be a positive integer.")
        if control and (not (isinstance(control, int) and control >= 0)):
            raise ValueError("Qubit index must be a positive integer.")

        self.__dict__ = {"name": name, "target": target, "control": control,
                          "parameter": parameter, "is_variational": is_variational}

    def __str__(self):
        """ Print gate information in a somewhat formatted way. Do not print empty attributes. """

        mystr = f"{self.name:<10}"
        for attr in ["target", "control", "parameter"]:
            if self.__getattribute__(attr) or isinstance(self.__getattribute__(attr), int):
                mystr += f"{attr} : {self.__getattribute__(attr)}   "
        if self.is_variational:
            mystr += "\t (variational)"

        return mystr
