"""Tools for performing Bravyi-Kitaev Transformation, as prescribed via the Fenwick Tree mapping.
This implementation accommodates mapping of qubit registers where the number of qubits is not a 
power of two."""

import itertools
import numpy

from openfermion.utils import count_qubits
from openfermion.ops import InteractionOperator, MajoranaOperator, DiagonalCoulombHamiltonian

from qsdk.toolboxes.operators import FermionOperator, QubitOperator