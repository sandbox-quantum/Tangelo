# Changelog

This file documents the main changes between versions of the code.

## [0.4.3] - 2024-05-21

### Added

- DMET: HF and MP2 solvers
- DMET: fragment active space can be specified by usrs as a callable function (see DMET notebook)

### Changed

- Copyrights (SandboxAQ 2024)
- Call to qiskit state vector simulator and IBM Q hardware experiment submissions (compatibility with Qiskit v1.0)


### Deprecated / Removed

## [0.4.2] - 2023-12-20

### Added

- iQPE algorithm
- support for adaptive circuit with mid-measurement controlled operations
- iFCI fragment import
- FNO for active space selection
- UHF symmetry reference labels
- IBMConnection now supports target instance


### Changed

- Performance improvement: VQE get_rdm
- Feature: trim qubits flag for circuit.split
- Bugfix: adapt to new qiskit version for noisy simulation
- Bugfix: DMET fix for solvers and fragment object initialization with options
- Bugfix: trim_trivial_qubits


### Deprecated / Removed


## [0.4.1] - 2023-10-18

### Added

- QM/MM problem decomposition
- QPE framework
- Truncated taylor series function returning qubits
- simplify method on Circuit

### Changed

- Automated testing currently covering python 3.8, 3.9, 3.10, 3.11 (#333)
- Installation: pyscf removed from requirements
- Performance improvement: combinatorial mapping
- Feature: ILC iteration now implements exact expansion / parameters
- Feature: VQE-like algorithms verbose mode now prints and tracks energies, for users interested in the convergence of the algorithm.
- Bugfix: Combinatorial mapping now handles spin != 0 (#330)
- Bugfix: get_expectation_value takes into account n_shots for all backends supporting the option.
- Bugfix: Fix corner case of FCI and CCSD solvers calculations (mo coefficients were occasionally recomputed differently).
- Bugfix: Updates in IBMConnection to keep up with changes in qiskit-runtime.


### Deprecated / Removed



## [0.4.0] - 2023-06-29

### Added

- Psi4 and pyscf optional dependencies, can be used as chemistry backends for classical calculations
- symbolic simulator
- stim clifford simulator
- Support for UHF reference mean-field in DMET
- trimming trivial qubits from Hamiltonians and circuits
- BraketConnection class
- combinatorial qubit mapping
- MP2Solver

### Changed

- Bugfix: DMET with virtual space truncation threshold, as well as ecp
- ADAPT now supports spin as parameter

### Deprecated / Removed

- in linq: Old translation functions, and Simulator class (use get_backend or translate_circuit instead)


## [0.3.4] - 2023-02-15

### Added

- Richardson extrapolation: support for variance and std dev (+ bugfix)
- Allow single flip index dis for QCC
- Notebook: iQCC using only Clifford circuits
- UHF reference mean-field
- Multi-product, grid circuits and discrete clock
- translate_circuit now supports Pennylane format and full bidirectional translation for all formats
- translate_op now supports all formats, both ways
- Saving mid-circuit measurements
- Selecting a desired mid-circuit measurement for simulation
- compute_rdms from a classical shadow, experimental data or using a backend on the fly
- support for frozen orbitals for each fragment in DMET
- Notebook for Tangelo + IBM Quantum demo (IBMQConnection)
- draw method for circuits

### Changed

- Bumped Python version number to 3.8 as 3.7 is no longer supported
- Bugfix: DMET + QCC
- Auto-threshold cutoff for small coefficients in LCU
- examples folder no longer in main repo, moved to Tangelo-Examples repo
- Bugfix: IBMQConnection (API update)

### Deprecated



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
