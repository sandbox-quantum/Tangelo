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

import functools
import os
import sys
import warnings
from importlib import util


class HiddenPrints:
    """Class to hide terminal printing with a 'with' block."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def is_package_installed(package_name):
    """Check if module is installed without importing."""
    spam_spec = util.find_spec(package_name)
    return spam_spec is not None


def deprecated(custom_message=""):
    """This is a decorator which can be called to mark functions as deprecated.
    It results in a warning being emitted when the function is used.

    Ref: https://stackoverflow.com/a/30253848

    Args:
        custom_message (str): Message to append to the deprecation warning.
    """
    def deprecated_wrapper(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Function {func.__name__} will be deprecated in a future release. {custom_message}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func
    return deprecated_wrapper


# List all backends and statevector ones supported by Simulator class
all_backends = {"qulacs", "qiskit", "cirq", "braket", "projectq"}
all_backends_simulator = {"qulacs", "qiskit", "cirq"}
sv_backends_simulator = {"qulacs", "qiskit", "cirq"}

# Dictionary mapping package names to their identifier in this module
packages = {p: p for p in all_backends}

# Figure out what is install in user's environment
installed_backends = {p_id for p_id, p_name in packages.items() if is_package_installed(p_name)}
installed_simulator = installed_backends & all_backends_simulator
installed_sv_simulator = installed_backends & sv_backends_simulator

# Check if qulacs installed (best performance for tests). If not, defaults to cirq (always installed with openfermion)
default_simulator = "qulacs" if "qulacs" in installed_sv_simulator else "cirq"
