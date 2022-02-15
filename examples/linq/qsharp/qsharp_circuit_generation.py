"""
This script shows how this package can be used to generate Q# code, which may run locally and be tested against
the QDK simulator or submitted to various devices through the Azure Quantum cloud services.

In particular, it is easy to write a circuit once and then generate variations of it
(parameter sweep, measurement basis...), avoiding us the trouble of duplicating things manually, which is also error-prone.
Below, an example with a simple "base" circuit of two CNOTS and a parameterized rotation.

Q# code needs to be written to file, as it needs to be compiled before execution, regardless of the compute backend.
"""

from tangelo.linq import Gate, Circuit
from tangelo.linq.translator import translate_qsharp
from tangelo.linq.helper_circuits import measurement_basis_gates, pauli_string_to_of


def theta_sweep(theta, m_basis):
    """ A single-parameter example circuit, with change of basis at the end if needed """
    my_gates = [Gate('CNOT', target=0, control=1),
                Gate('RX', target=1, parameter=theta),
                Gate('CNOT', target=0, control=1)]
    my_gates += measurement_basis_gates(pauli_string_to_of(m_basis))
    return Circuit(my_gates)


for theta, m_basis in [(0.1, 'ZZ'), (0.2, 'ZZ'), (0.3, 'XY')]:

    # Instantiate abstract circuit with correct theta and measurement basis
    c = theta_sweep(theta, m_basis)
    print(f"\n{c}")

    # Translate to Q#
    qc = translate_qsharp(c)

    # Write to file
    with open(f'theta_sweep_{str(theta)}_{m_basis}.qs', 'w+') as of:
        of.write(qc)
