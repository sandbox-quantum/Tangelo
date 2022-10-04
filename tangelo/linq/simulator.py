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

"""Simulator Class, encompases various simulators that abstract their
differences from the user. Returns a class that Able to run noiseless and noisy simulations,
leveraging the capabilities of different backends, quantum or classical.

If the user provides a noise model, then a noisy simulation is run with n_shots
shots. If the user only provides n_shots, a noiseless simulation is run, drawing
the desired amount of shots. If the target backend has access to the statevector
representing the quantum state, we leverage any kind of emulation available to
reduce runtime (directly generating shot values from final statevector etc) If
the quantum circuit contains a MEASURE instruction, it is assumed to simulate a
mixed-state and the simulation will be carried by simulating individual shots
(e.g a number of shots is required).

Some backends may only support a subset of the above. This information is
contained in the static method backend_info for each backend.
"""

from typing import Union, Tuple, Type

import numpy

from tangelo.linq.simulator_base import SimulatorBase
from tangelo.linq.target import target_dict, backend_info
from tangelo.linq.noisy_simulation import NoiseModel
from tangelo.helpers.utils import default_simulator


class Simulator(SimulatorBase):

    def __new__(self, target=default_simulator, n_shots=None, noise_model=None, **kwargs):
        """Return requested target simulator class

        Args:
            target (string or Type[SimulatorBase]): String can be "qiskit", "cirq", "qdk" or "qulacs". Can also provide
                a subclass of SimulatorBase.
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model (NoiseModel): A noise model object assumed to be in the format
                expected from the target backend.
            kwargs: Other arguments that could be passed to a target. Examples are qubits_to_use for a QPU, transpiler optimization
                level, error mitigation flags etc.
        """

        # Return new instance of target class
        # If target is a string use target_dict to return built-in Target Simulators
        if isinstance(target, str):
            return target_dict[target](n_shots=n_shots, noise_model=noise_model)
        # Try to instantiate the target as a Simulator
        elif issubclass(target, SimulatorBase):
            return target(n_shots=n_shots, noise_model=noise_model, **kwargs)
        else:
            raise ValueError("target must be a str or a subclass of SimulatorBase.")

    def __init__(self, target: Union[str, Type[SimulatorBase]] = default_simulator, n_shots: Union[None, int] = None,
                 noise_model: Union[None, NoiseModel] = None, **kwargs):
        """Return requested target simulator class

        Args:
            target (string or Type[SimulatorBase]): String can be "qiskit", "cirq", "qdk" or "qulacs". Can also provide
                a subclass of SimulatorBase.
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model (NoiseModel): A noise model object assumed to be in the format
                expected from the target backend.
            kwargs: Other arguments that could be passed to a target. Examples are qubits_to_use for a QPU, transpiler optimization
                level, error mitigation flags etc.
        """
        pass

    def simulate_circuit(self) -> Union[Tuple[dict, numpy.ndarray], Tuple[dict, None]]:
        pass

    @staticmethod
    def backend_info() -> dict:
        pass
