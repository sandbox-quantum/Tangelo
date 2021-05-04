# QEMIST_qSDK contents

This repository contains two separate packages, called `qSDK` and `agnostic_simulator`.
Despite living in the same repository, these two packages are different entities. Their respective code, documentation,
tutorials, and installation instructions are contained in their own dedicated folder. Please refer to them.

- `qSDK` provides access to the tools developed for quantum chemistry simulation on quantum computers and emulators. 
The package provides access to some problem decomposition techniques, electronic structure solvers, as well as the
various toolboxes containing the functionalities necessary to build these workflows, or your own.
It was developed to be compatible with QEMIST Cloud, to seamlessly enable the use of large scale problem decomposition
combined with both classical and quantum electronic structure solvers in the cloud. The package also provides local
problem decomposition techniques, and is designed to support the output of a problem decomposition method performed
through QEMIST Cloud as well.

- `agnostic_simulator` can be thought to be a quantum circuit simulation engine, allowing users to target a variety
of compute backends (QPUs and simulators) available on their local machine or through cloud services. In itself,
  it has nothing to do with quantum chemistry, and thus has a wide range of applicability. `qSDK` relies on `agnostic_simulator`
  for quantum circuit simulation, and requires that this package is installed.
  