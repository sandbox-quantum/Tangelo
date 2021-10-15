"""
    Example of how to process IonQ QPU results to compute some expectation values.
    This example assumes that a QPU experiment was run on a IonQ device and that a histogram of results was returned.

    The histogram is first translated in standard format, and backendbuddy is then used to compute an
    expectation values using these frequencies.

    If you wish to compute theoretical value using a simulator, please write your circuit in abstract format and
    instantiate the Simulator class with a backend such as Qiskit, ProjectQ or qulacs, and just call the
    "get_expectation_value" method. There is zero benefit in using the IonQ simulator, it does exactly the same.
"""

from qsdk.backendbuddy import Simulator
from qsdk.backendbuddy.helpers import pauli_string_to_of

# Example of IonQ QPU results written in a Python-friendly format
# The dictionary below contains 5 histograms corresponding to 5 different experiments, each for a different basis
QPU_results = {'XX': {'0': 0.2134, '1': 0.2935, '2': 0.2886, '3': 0.2045},
               'XZ': {'0': 0.0054, '1': 0.0055, '2': 0.5069, '3': 0.4822},
               'YY': {'1': 0.2132, '0': 0.291, '3': 0.2965, '2': 0.1993},
               'ZX': {'0': 0.0132, '1': 0.4906, '2': 0.0068, '3': 0.4894},
               'ZZ': {'0': 0.01, '1': 0.0054, '2': 0.0174, '3': 0.9672}}
n_qubits = 2


def dd(d, n_qubits):
    """ Remaps IonQ histogram keys to binary bit strings corresponding to classical state (q0q1...qn) """
    new_d = dict()
    for k, v in d.items():
        bs = bin(int(k)).split('b')[-1]
        new_k = "0" * (n_qubits - len(bs)) + bs
        new_d[new_k[::-1]] = v
    return new_d


def pauli_string_to_of_string(ps):
    """ Turns strings of type XXIZ into strings like X0 X1 Z3, for convenience with expectation value functions """
    ofs = ''
    for i, l in enumerate(ps):
        if l != 'I':
            ofs += (l + str(i) + ' ')
    return ofs[:-1]


# Compute all experimental expectation values
# -------------------------------------------
QPU_results = {term: dd(hist, n_qubits) for term, hist in QPU_results.items()}

# Use experiment results as you see fit to compute your expectation values of interest
for term, hist in QPU_results.items():
    print(f'{term}  {Simulator.get_expectation_value_from_frequencies_oneterm(term=pauli_string_to_of(term), frequencies=hist):.5f}')
print(f'ZI  {Simulator.get_expectation_value_from_frequencies_oneterm(term=pauli_string_to_of("ZI"), frequencies=QPU_results["ZZ"]):.5f}')
print(f'IZ  {Simulator.get_expectation_value_from_frequencies_oneterm(term=pauli_string_to_of("IZ"), frequencies=QPU_results["ZZ"]):.5f}')
print(f'IX  {Simulator.get_expectation_value_from_frequencies_oneterm(term=pauli_string_to_of("IX"), frequencies=QPU_results["ZX"]):.5f}')
print(f'XI  {Simulator.get_expectation_value_from_frequencies_oneterm(term=pauli_string_to_of("XI"), frequencies=QPU_results["XZ"]):.5f}')
