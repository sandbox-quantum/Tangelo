import unittest
import os
from openfermion import load_operator

from tangelo.linq import translator, Simulator, Circuit
from tangelo.helpers import string_ham_to_of, measurement_basis_gates
from tangelo.toolboxes.measurements import group_qwc, exp_value_from_measurement_bases

path_data = os.path.dirname(os.path.abspath(__file__)) + '/data'

def test_sorted_qubitwise_commutativity_of_H2():
    """ The JW Pauli hamiltonian of H2 at optimal geometry is a 15-term operator. Using qubitwise-commutativity,
    it is possible to get the expectation value of all 15 terms by only performing measurements in 5 distinct
    measurement bases. This test first verifies the 5 measurement bases have been identified, and then derives
    the expectation value for the qubit Hamiltonian.
    """

    # Load qubit Hamiltonian
    qb_ham = load_operator("mol_H2_qubitham.data", data_directory=path_data, plain_text=True)

    # Group Hamiltonian terms using qubitwise commutativity
    sorted_grouped_ops = group_sorted_qwc(qb_ham)

    # Load an optimized quantum circuit (UCCSD) to compute something meaningful in this test
    with open(f"{path_data}/H2_UCCSD.qasm", "r") as f:
        openqasm_circ = f.read()
    abs_circ = translator._translate_openqasm2abs(openqasm_circ)

    # Only simulate and measure the wavefunction in the required bases (simulator or QPU), store in dict.
    histograms = dict()
    sim = Simulator()
    for basis, sub_op in sorted_grouped_ops.items():
        full_circuit = abs_circ + Circuit(measurement_basis_gates(basis))
        histograms[basis], _ = sim.simulate(full_circuit)

    # Reconstruct exp value of initial input operator using the histograms corresponding to the suboperators
    exp_value = exp_value_from_measurement_bases(group_sorted_qwc, histograms)
    print(exp_value, "/n", sim.get_expectation_value(qb_ham, abs_circ))

def group_sorted_qwc(op):
    """
    Args:
        op (QubitOperator): qubit operator

    Returns:
        commutativity groups (list): groupings of one, "master" pauli word that is connected to a coefficient
    """
    # sorts terms on coefficient in decreasing order
    terms = op.terms
    terms = [(k, v) for k, v in sorted(terms.items(), key=lambda x: abs(x[1]), reverse=True)]

    sorted_qwc_groups = []  # {(Word): (Qubit Operator w Terms)}

    for pauli_word, coeff in terms:
        commutes = False
        for index, master in enumerate(sorted_qwc_groups):
            if does_commute(master[0], pauli_word):
                synthesized = tuple(set.union(set(master[0]), set(pauli_word)))  # Sort terms on qubits
                sorted_qwc_groups[index] = [synthesized, max(abs(master[1]), abs(coeff))]
                commutes = True
                break
        if not commutes:
            sorted_qwc_groups.append([pauli_word, coeff])

    return sorted_qwc_groups

def does_commute(pauli_1, pauli_2):
    """
    Args:
        pauli_1 (tuple): pauli word not including coefficient
        pauli_2 (tuple): pauli word not including coefficient

    Returns:
        commutativity (bool): returns whether the two pauli words qubit-wise commute
    """
    pauli_1 = dict(pauli_1)
    pauli_2 = dict(pauli_2)

    for key, value in pauli_1.items():
        if key not in pauli_1.keys(): continue
        try:
            if pauli_1[key] == pauli_2[key]: continue
        except KeyError:
            continue

        return False
    return True

test_sorted_qubitwise_commutativity_of_H2()
