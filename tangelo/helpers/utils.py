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

"""This file provides information about what the backends currently installed in
the user's environment, for the purpose of running / skipping tests, and setting
a default simulator.
"""

import os
import sys


class HiddenPrints:
    """Class to hide terminal printing with a 'with' block."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_package_installed(package_name):
    try:
        exec(f"import {package_name}")
        # DEBUG print(f'{package_name:16s} :: found')
        return True
    except ModuleNotFoundError:
        # DEBUG print(f'{package_name:16s} :: not found')
        return False


# List all backends and statevector ones supported by Simulator class
all_backends = {"qulacs", "qiskit", "cirq", "braket", "projectq", "qdk"}
all_backends_simulator = {"qulacs", "qiskit", "cirq", "qdk"}
sv_backends_simulator = {"qulacs", "qiskit", "cirq"}

# Dictionary mapping package names to their identifier in this module
packages = {p: p for p in all_backends}
packages["qdk"] = "qsharp"

# Figure out what is install in user's environment
installed_backends = {p_id for p_id, p_name in packages.items() if is_package_installed(p_name)}
installed_simulator = installed_backends & all_backends_simulator
installed_sv_simulator = installed_backends & sv_backends_simulator

# Check if qulacs installed (best performance for tests). If not, defaults to cirq (always installed with openfermion)
default_simulator = "qulacs" if "qulacs" in installed_sv_simulator else "cirq"
