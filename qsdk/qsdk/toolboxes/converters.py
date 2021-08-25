""" Module defining helper functions to juggle with different data structures. """

from qsdk.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from  qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.operators.operators import QubitHamiltonian, count_qubits


def molecule_to_secondquantizedmolecule(mol, basis_set="sto-3g", frozen_orbitals=0):
    """ Function to convert a Molecule into a SecondQuantizedMolecule.

        Args:
            mol (Molecule): Self-explanatory.
            basis_set (string): String representing the basis set.
            frozen_orbitals (int or list of int): Number of MOs or MOs indexes
                to freeze.

        Returns:
            SecondQuantizedMolecule: Mean-field data structure for a molecule.
    """

    converted_mol = SecondQuantizedMolecule(mol.xyz, mol.q, mol.spin,
                                            basis=basis_set,
                                            frozen_orbitals=frozen_orbitals)
    return converted_mol

def secondquantizedmolecule_to_qubithamiltonian(mol, mapping="JW", up_then_down=False):
    """ Function to convert a SecondQuantizedMolecule into a QubitHamiltonian.

        Args:
            mol (SecondQuantizedMolecule): Self-explanatory.
            mapping (string): Qubit mapping procedure.
            up_then_down (bool): Whether or not spin ordering is all up then
                all down.

        Returns:
            QubitHamiltonian: Self-explanatory.
    """
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
    """ Function to convert a QubitOperator into a QubitHamiltonian.

        Args:
            qubit_op (QubitOperator): Self-explanatory.
            n_qubits (int): Self-explanatory.
            mapping (string): Qubit mapping procedure.
            up_then_down (bool): Whether or not spin ordering is all up then
                all down.

        Returns:
            QubitHamiltonian: Self-explanatory.
    """
    qubit_ham = QubitHamiltonian(n_qubits, mapping, up_then_down)

    for term, coeff in qubit_op.terms.items():
        qubit_ham += QubitHamiltonian(n_qubits, mapping, up_then_down, term, coeff)

    return qubit_ham
