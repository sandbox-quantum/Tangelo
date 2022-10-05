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

"""Class acting as a unified interface able to instantiate any simulator object
associated to a specific target (qulacs, qiskit, cirq, user-defined...). Some
built-in target simulators support features such as noisy simulation, the ability
to run noiseless simulation with shots, or particular emulation techniques. Target
backends can also be implemented using APIs to quantum devices.
"""

from typing import Union, Type

from tangelo.linq.simulator_base import SimulatorBase
from tangelo.linq.target import target_dict, backend_info
from tangelo.linq.noisy_simulation import NoiseModel
from tangelo.helpers.utils import default_simulator


class Simulator(SimulatorBase):
    """Class that when initialized becomes an instance of a target subclass of SimulatorBase. All available targets can be
    found in tangelo.linq.target. Can also be used to access static methods of SimulatorBase without initializing."""

    def __init__(self, target: Union[None, str, Type[SimulatorBase]] = default_simulator, n_shots: Union[None, int] = None,
                 noise_model=None, **kwargs):
        """Initialize requested target simulator object.

        Args:
            target (string or Type[SimulatorBase]): String can be "qiskit", "cirq", "qdk" or "qulacs". Can also provide
                a subclass of SimulatorBase.
            n_shots (int): Number of shots if using a shot-based simulator.
            noise_model: A noise model object assumed to be in the format expected from the target backend.
            kwargs: Other arguments that could be passed to a target. Examples are qubits_to_use for a QPU, transpiler
                optimization level, error mitigation flags etc.
        """

        if target is None:
            target = default_simulator
        # If target is a string use target_dict to return built-in Target Simulators
        if isinstance(target, str):
            target = target_dict[target]
        # If subclass of SimulatorBase, use target
        elif issubclass(target, SimulatorBase):
            pass
        else:
            raise TypeError(f"target must be a str or a subclass of SimulatorBase but received class {type(target).__name__}")
        # Update class variables with __dict__ coming from initialized target class.
        self.__class__ = target
        self.__dict__.update(target(n_shots=n_shots, noise_model=noise_model, **kwargs).__dict__)

    def simulate_circuit(self):
        pass

    @staticmethod
    def backend_info() -> dict:
        pass
