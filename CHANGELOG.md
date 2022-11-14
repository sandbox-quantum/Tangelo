# Changelog

This file documents the main changes between versions of the code.


## [0.3.3] - 2022-11-09

### Added

- Circuit translation from any supported source to any supported target format, with a single function
- Translation for qubit / Pauli operators for qiskit format
- All algorithms now run with any built-in or user-defined backend, simulator or QPU.
- TETRIS-ADAPT VQE
- iQCC-ILC
- Quantum signal processing time-evolution
- Higher even-order trotterization for time-evolution
- Histogram class, featuring methods for renormalization, post-selection, aggregation
- Computation of variance of expectation values
- Function to compute RDMs from experimental data / classical shadow
- IBMConnection Class for submission of experiments to IBM Quantum
- qchem_modelling_basics and excited_states notebooks

### Changed

- All notebooks now launchable with Google Collab
- Docker image updated

### Deprecated

- Simulator class deprecated in favor of get_backend function in linq
- backend-specific translate_xxx functions (e.g translate_qiskit, translate_qulacs...) deprecated in favor of translate_circuit in linq


## [0.3.2] - 2022-08-06

### Added

- Linear Combination of Unitaries (LCU)
- QEMIST Cloud MI-FNO innregration: interface adjustments
- iQCC ansatz for VQE
- IonQConnection class and notebook, to facilitate experiments through IonQ's API
- FCISolver active space selection / frozen orbitals: restrictions for half-empty orbitals

### Changed

- QEMIST Cloud MI-FNO innregration: interface adjustments
- ADAPT-VQE interface: spin only required when needed
- VQE returns warning or error if no variational parameters are in the ansatz circuit
- Bug fix: scBK reference state, HEA ansatz initialization
- Check for valid number of target qubits for common gates
- Documentation, README

### Deprecated

## [0.3.1] - 2022-06-15

### Added

- Depth method for circuits, gate cancellation methods for simple optimisations
- QCC-ILC and iQCC solver
- Support for MI-FNO fragments coming from QEMIST Cloud
- ONIOM notebook
- Quantum deflation
- SA-VQE solver, SA-OO-VQE solver
- HybridOperator for speedup for QubitOperator on certain operations in stabilizer notation
- Support for symmetry in pyscf computations

### Changed

- DMET recomputes mean-field when working with atom indices, to fix bug.
- Documentation, README, CONTRIBUTIONS

### Deprecated


## [0.3.0] - 2022-02-15

### Added

- Circuit operators and methods (repetition, equality, trim, split, stack...)
- Support for Classical Shadows ((de)randomized, adaptative)
- Sphinx documentation generator script in dev_tools
- JKMN qubit mapping
- QMF, QCC and VSQS ansatze for VQE
- Controlled-time evolution, Richardson extrapolation method

### Changed

- Naming (Good Chemistry, Tangelo, linq)

### Deprecated
