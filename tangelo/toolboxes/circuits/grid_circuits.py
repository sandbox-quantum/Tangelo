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

"""Module to generate the circuits for grid based computation"""
from typing import Union, List

import numpy as np

from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit
from tangelo.linq import Gate, Circuit


def get_xsquared_circuit(dt: float, dx: float, fac: float, x0: float, delta: float,
                         qubit_list: List[int], control: Union[None, int, List[int]] = None) -> Circuit:
    """Return circuit for exp(-1j*dt*[fac*(x - x0)**2 + delta]) as defined in arXiv:2006.09405

    Args:
        dt (float): Time to evolve.
        dx (float): Grid spacing.
        fac (float): Factor in front of x^2 term.
        x0 (float): Shift for (x-x0)^2 term
        delta (float): Constant shift
        qubit_list (List[int]): Qubits to apply circuit to. The order is important depending on lsq_first or msq_first
        control (Union[int, List[int]]): The control qubits

    Returns:
        Circuit: The circuit that applies exp(-1j*dt*[fac*(x-x0)**2 +delta])
    """

    if control is not None:
        gate_name = 'CPHASE'
        clist = [control] if isinstance(control, (int, np.integer)) else control
        clist2 = clist
    else:
        gate_name = 'PHASE'
        clist = None
        clist2 = []

    gate_list = []

    # Constant terms
    prefac = -dt*(fac*x0**2 + delta)
    gate_list.append(Gate(gate_name, target=qubit_list[0], parameter=prefac, control=clist))
    gate_list.append(Gate('X', target=qubit_list[0]))
    gate_list.append(Gate(gate_name, target=qubit_list[0], parameter=prefac, control=clist))
    gate_list.append(Gate('X', target=qubit_list[0]))

    # Linear terms
    prefac = 2*dt*fac*x0*dx
    for i, q in enumerate(qubit_list):
        gate_list.append(Gate(gate_name, target=q, parameter=prefac*2**i, control=clist))

    # Quadratic terms
    prefac = -dt*fac*dx**2
    for i, q1 in enumerate(qubit_list):
        gate_list.append(Gate(gate_name, target=q1, parameter=prefac*2**(2*i), control=clist))
        for j, q2 in enumerate(qubit_list):
            if (i != j):
                gate_list.append(Gate('CPHASE', control=[q1]+clist2, target=q2, parameter=prefac*2**(i+j)))

    return Circuit(gate_list)


def get_psquared_circuit(dt: float, dx: float, mass: float, qubit_list: List[int],
                         control: Union[int, List[int]] = None) -> Circuit:
    """Return circuit for p^2/2/m as defined in arXiv:2006.09405 using qft

    Args:
        dt (float): Time to evolve.
        dx (float): Grid spacing.
        mass (float): The mass used for the time-evolution.
        qubit_list (List[int]): Qubits to apply circuit to. The order is important depending on lsq_first or msq_first
        control (Union[int, List[int]]): The control qubits

    Returns:
        Circuit: The circuit that applies exp(-1j*dt*p^2/2/m)
    """
    n_b = 2**len(qubit_list)
    dp = 2*np.pi/n_b/dx
    p0 = n_b//2*dp
    flip_gate = Circuit([Gate('X', target=qubit_list[-1])])
    circuit = Circuit()
    circuit += flip_gate + get_qft_circuit(qubit_list, swap=True) + flip_gate
    circuit += get_xsquared_circuit(dt, dp, 1/2/mass, p0, 0, qubit_list, control=control)
    circuit += flip_gate + get_qft_circuit(qubit_list, inverse=True, swap=True) + flip_gate
    return circuit
