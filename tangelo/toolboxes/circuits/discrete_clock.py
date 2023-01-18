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

"""Module to generate the circuits necessary to implement discrete clock time
Refs:
    [1] Jacob Watkins, Nathan Wiebe, Alessandro Roggero, Dean Lee, "Time-dependent Hamiltonian
    Simulation using Discrete Clock Constructions" arXiv: 2203.11353
"""
import math
from typing import List, Callable

import numpy as np

from tangelo.linq import Circuit, Gate
from tangelo.toolboxes.circuits.multiproduct import get_multi_product_circuit
from tangelo.toolboxes.ansatz_generator.ansatz_utils import get_qft_circuit


def get_adder_circuit(qubit_list: List[int], t: int) -> Circuit:
    """Return circuit that takes all bitstrings and add binary(t) to it.

    Args:
        qubit_list (List[int]): The qubits to apply the addition of t.
        t (int): The integer to add

    Returns:
        Circuit: The circuit that applies the addition of t"""

    flip_gate = Circuit([Gate("X", qubit_list[-1])])
    fft = flip_gate + get_qft_circuit(qubit_list, swap=True) + flip_gate
    ifft = flip_gate + get_qft_circuit(qubit_list, inverse=True, swap=True) + flip_gate

    gate_list = []
    for i, q in enumerate(qubit_list):
        gate_list.append(Gate('PHASE', target=q, parameter=2*np.pi*t*2**i/2**len(qubit_list)))

    return fft+Circuit(gate_list)+ifft


def get_discrete_clock_circuit(trotter_func: Callable[..., Circuit], trotter_kwargs: dict, n_state_qus: int,
                               time: float, n_time_steps: int, mp_order: int) -> Circuit:
    """Return discrete clock circuit as described in arXiv: 2203.11353

    Args:
        trotter_func (Callable[..., Circuit]): The function that implements the controlled 2nd order trotter time-evolution
            starting at "t0" for "time" using "n_trotter_steps" using "control"
        trotter_kwargs (dict): Other keyword arguments for trotter_func.
        n_state_qus (int): The number of qubits used to represent the state to time-evolve.
        time (float): The total time to evolve.
        n_time_steps (int): The number of time steps in the discrete clock.
        mp_order (int): The multi-product order to use for the time-evolution.

    Returns:
        Circuit: The time-evolution circuit using the discrete clock construction.
    """

    circuit = Circuit()
    n_mp_qus = math.ceil(np.log2(mp_order+2))
    n_fft_qus = math.ceil(np.log2(n_time_steps))
    fft_start = n_state_qus+n_mp_qus
    fft_qus = list(reversed(range(fft_start, fft_start+n_fft_qus)))

    dt = time/n_time_steps

    for i in range(n_time_steps):
        birep = np.binary_repr(i, width=n_fft_qus)
        x_ladder = Circuit([Gate("X", c+fft_start) for c, j in enumerate(birep) if j == "0"])
        circuit += x_ladder
        trotter_kwargs['t0'] = i*dt
        trotter_kwargs['time'] = dt
        circuit += get_multi_product_circuit(dt, mp_order, n_state_qus, control=fft_qus,
                                             second_order_trotter=trotter_func, trotter_kwargs=trotter_kwargs)
        circuit += x_ladder + get_adder_circuit(fft_qus, 1)

    circuit += get_adder_circuit(fft_qus, -n_time_steps)

    return circuit
