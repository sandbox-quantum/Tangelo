"""Docstring. """

from qsdk.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from  qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.operators.operators import QubitHamiltonian, count_qubits

def molecule_to_secondquantizedmolecule(mol, basis_set="sto-3g", frozen_orbitals=0):
    """Docstring. """

    converted_mol = SecondQuantizedMolecule(mol.xyz, mol.q, mol.spin,
                                            basis=basis_set,
                                            frozen_orbitals=frozen_orbitals)
    return converted_mol

def secondquantizedmolecule_to_qubithamiltonian(mol, mapping="JW", up_then_down=False):
    """Docstring. """

    n_spinorbitals = mol.n_active_sos
    n_electrons = mol.n_active_electrons

    qubit_op = fermion_to_qubit_mapping(mol.fermionic_hamiltonian,
                                        mapping,
                                        n_spinorbitals,
                                        n_electrons,
                                        up_then_down)

    n_qubits = count_qubits(qubit_op)
    qubit_ham = qubitop_to_qubitham(qubit_op, n_qubits, mapping, up_then_down)

    return qubit_ham

def qubitop_to_qubitham(qubit_op, n_qubits, mapping, up_then_down):
    """Docstring. """

    qubit_ham = QubitHamiltonian(n_qubits, mapping, up_then_down)

    for term, coeff in qubit_op.terms.items():
        qubit_ham += QubitHamiltonian(n_qubits, mapping, up_then_down, term, coeff)

    return qubit_ham
