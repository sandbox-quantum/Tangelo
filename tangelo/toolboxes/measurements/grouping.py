import unittest
import os
from openfermion import load_operator

from tangelo.linq import translator, Simulator, Circuit
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.helpers import measurement_basis_gates
from tangelo.toolboxes.measurements import exp_value_from_measurement_bases

path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'



