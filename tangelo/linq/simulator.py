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

"""Factory function acting as a unified interface able to instantiate any simulator object
associated to a specific target (qulacs, qiskit, cirq, user-defined...). Some
built-in target simulators support features such as noisy simulation, the ability
to run noiseless simulation with shots, or particular emulation techniques. Target
backends can also be implemented using APIs to quantum devices.
"""

from typing import Union, Type

from tangelo.linq.target.backend import Backend
from tangelo.linq.target import target_dict
from tangelo.helpers.utils import default_simulator, deprecated


def get_backend(target: Union[None, str, Type[Backend]] = default_simulator, n_shots: Union[None, int] = None,
                noise_model=None, **kwargs):
    """Return requested target backend object.

    Args:
        target (string or Type[Backend] or None): Supported string identifiers can be found in
            target_dict (from tangelo.linq.target). Can also provide a user-defined backend (child to Backend class)
        n_shots (int): Number of shots if using a shot-based simulator.
        noise_model: A noise model object assumed to be in the format expected from the target backend.
        kwargs: Other arguments that could be passed to a target. Examples are qubits_to_use for a QPU, transpiler
            optimization level, error mitigation flags etc.
     """

    if target is None:
        target = target_dict[default_simulator]
    # If target is a string use target_dict to return built-in backend
    elif isinstance(target, str):
        try:
            target = target_dict[target]
        except KeyError:
            raise ValueError(f"Error: backend {target} not supported. Available built-in options: {list(target_dict.keys())}")
    elif not issubclass(target, Backend):
        raise TypeError(f"Target must be a str or a subclass of Backend but received class {type(target).__name__}")

    return target(n_shots=n_shots, noise_model=noise_model, **kwargs)
