"""
    This file provides information about what the backends currently installed in the user's environment,
    for the purpose of running / skipping tests, and setting a default simulator.
"""


def is_package_installed(package_name):
    try:
        exec(f'import {package_name}')
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
