# Changelog

This file documents the main changes between versions of the code.


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
