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

"""Module to generate the circuits necessary to implement multi-product time-evolution
Refs:
    [1] Guang Hao Low, Vadym Kliuchnikov and Nathan Wiebe, "Well-conditioned multi-product
    Hamiltonian simulation" arXiv: 1907.11679
"""

import math
from typing import Union, Tuple, List, Callable

import numpy as np

from tangelo.linq import Circuit, Gate
from tangelo.linq.helpers.circuits.statevector import StateVector
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize
from tangelo.toolboxes.circuits.lcu import sign_flip
from tangelo.toolboxes.operators import QubitOperator, FermionOperator


def get_ajs_kjs(order: int) -> Tuple[List[float], List[int], int]:
    """Return aj coefficients and number of steps kj for multi-product order
    The first two indices of aj coefficients are the portion need to make the one-norm sum to two.

    Args:
        order (int): The desired order of expansion

    Returns:
        List[float], List[int], int: aj coefficients, kj steps, number of ancilla qubits needed
    """

    if not isinstance(order, (int, np.integer)):
        raise TypeError("order must be of integer type")
    if order < 1 or order > 6:
        raise ValueError("Tangelo currently only supports orders between 1 and 6")

    mp_qus = math.ceil(np.log2(order+2))

    adict = {1: [1],
             2: [1/3, 4/3],
             3: [1/105, 1/6, 81/70],
             4: [1/2376, 2/45, 729/3640, 31250/27027],
             5: [1/165888, 256/89775, 6561/179200, 390625/2128896, 6975757441/6067353600],
             6: [1/5544000, 8/19665, 81/4480, 65536/669375, 216/875, 7626831723/6537520000]}
    kdict = {1: [1],
             2: [1, 2],
             3: [1, 2, 6],
             4: [1, 2, 3, 10],
             5: [1, 2, 3, 5, 17],
             6: [1, 2, 3, 4, 6, 21]}
    fac = sum(adict[order])
    vlen = 2**mp_qus
    ajs = np.sqrt(np.abs([(2 - fac)/2, (2 - fac)/2] + [0]*(vlen-2-order) + adict[order]))
    ajs /= np.linalg.norm(ajs)
    kjs = [0]*(vlen-order) + kdict[order]
    return list(ajs), kjs, mp_qus


def get_multi_product_circuit(time: float, order: int, n_state_qus: int,
                              operator: Union[None, QubitOperator, FermionOperator] = None,
                              control: Union[int, list] = None,
                              second_order_trotter: Union[None, Callable[..., Circuit]] = None,
                              trotter_kwargs: Union[dict, None] = None) -> Circuit:
    """Return multi-product circuit as defined in arXiv: 1907.11679. Only up to 6th order is currently supported

    Args:
        time (float): The time to evolve.
        order (int): The order of the multi-product expansion
        n_state_qus (int): The number of qubits in the state to evolve.
        operator (Union[QubitOperator, FermionOperator]): The operator to evolve in time. Default None
        control (Union[int, List[int]]): The control qubit(s). Default None
        second_order_trotter (Callable[..., Circuit]): The callable function that defines the controlled 2nd order
            time-evolution. Must have arguments "control" and "n_trotter_steps".
        trotter_kwargs (dict): Other keyword arguments necessary to evaluate second_order_trotter.

    Returns:
        Circuit: The circuit representing the time-evolution using the multi-product construction.
    """

    if second_order_trotter is None:
        if operator is None:
            raise ValueError("Must supply second_order_trotter function or operator.")
        second_order_trotter = trotterize
        if trotter_kwargs is None:
            trotter_kwargs = {"operator": operator, "time": time, "trotter_order": 2}

    if control is not None:
        cont_list = control if isinstance(control, list) else [control]
    else:
        cont_list = []

    ajs, kjs, n_mp_qus = get_ajs_kjs(order)
    prep_qus = list(range(n_state_qus, n_state_qus+n_mp_qus))
    prep_state = StateVector(ajs, order="lsq_first")
    prep_circ = prep_state.initializing_circuit()
    prep_circ.reindex_qubits(prep_qus)

    ctrott = prep_circ
    for ii in range(2**n_mp_qus-1, 2**n_mp_qus-order-1, -1):
        birep = np.binary_repr(ii, width=n_mp_qus)
        x2_ladder = Circuit([Gate("X", c+n_state_qus) for c, j in enumerate(birep) if j == "0"])
        ctrott += x2_ladder
        ctrott += second_order_trotter(control=cont_list+prep_qus, n_trotter_steps=kjs[ii], **trotter_kwargs)
        # Add -1 phase for every other term in multi-product expansion
        if ii % 2 == 0:
            ctrott += Circuit([Gate("CRZ", 0, parameter=2*np.pi, control=cont_list+prep_qus)])
        ctrott += x2_ladder

    # add -I term for oblivious amplitude amplification
    birep = np.binary_repr(0, width=n_mp_qus)
    x2_ladder = Circuit([Gate("X", c+n_state_qus) for c, j in enumerate(birep) if j == "0"])
    ctrott += x2_ladder + Circuit([Gate("CRZ", 0, parameter=2*np.pi, control=cont_list+prep_qus)]) + x2_ladder

    ctrott += prep_circ.inverse()

    flip = sign_flip(prep_qus, control=cont_list)
    oaa_mp_circuit = ctrott + flip + ctrott.inverse() + flip + ctrott
    if control is not None:
        oaa_mp_circuit += Circuit([Gate("CRZ", 0, control=control, parameter=-2*np.pi)])

    return oaa_mp_circuit
