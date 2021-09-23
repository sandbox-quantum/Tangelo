# QEMIST_qSDK

Quantum SDK providing access to the tools developed for quantum chemistry simulation on quantum computers and emulators.

The package provides access to some problem decomposition techniques, electronic structure solvers, as well as the
various toolboxes containing the functionalities necessary to build these workflows, or your own.
The `backendbuddy` submodule allows users to target a variety
of compute backends (QPUs and simulators) available on their local machine or through cloud services.


It was developed to be compatible with QEMIST Cloud, to seamlessly enable the use of large scale problem decomposition
combined with both classical and quantum electronic structure solvers in the cloud. The package also provides local
problem decomposition techniques, and is designed to support the output of a problem decomposition method performed
through QEMIST Cloud as well.

## Contents of the repository

This folder contains the following files and subfolders:

- **qsdk** :
Code implementing various functionalities and algorithms to research and perform quantum chemistry experiments on
  gate-model quantum computers or simulators. The various features are exposed in toolboxes, and can be assembled to
  form workflows: electronic structure solvers.

- **docs** :
Source code documentation and user documentation (readthedocs)

- **examples** :
Examples and tutorials to learn how to use the different functionalities of the package
  

## Install

### 1.Prerequisites

You need to have Python3 installed (Python 2 is not supported anymore as of Jan 1st 2020, and has been officially replaced by Python 3)
in your environment. We recommend you use Python virtual environments (`virtualenv`) in order to set up your environment safely and cleanly, 
in the following. Brief overviews of virtualenv are available online, such as: https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv

We heavily recommend you have a good C/C++ compiler supporting modern C++ and OpenMP multithreading installed, 
such as the GNU ones (gcc, g++), as well as your own BLAS libraries. Your OS's package manager should provide a 
straightforward way to install these. This will have a big impact on the execution time of your code.

> :warning: You may encounter "python.h file not found" errors: ensure that you have the "dev" version of python3
> installed in your environment. Installation instructions may differ depending on your operating system, but are generally
> a one-liner using your distribution's package manager.

### 2.a Using pip

TODO: once this package is available on pypy, mention it here, give the command.

### 2.b From source, using setuptools

This package can be installed by first cloning this repository with `git clone`.
You can then install qSDK by using the setuptools with the `setup.py` file present in this very directory:
```
python setup.py install
```

For all information or trouble regarding `agnostic_simulator`, we recommend you check the code and documentation available in its 
dedicated folder. A few tips below:

> :warning: If you are using MacOS, your C/C++ compiler may be Clang, which does not support compilation of OpenMP
> multithreaded code. As a consequence, you may encounters errors, or see noticeable degradation in performance for some
> of the dependencies of this package. We recommend you install a suitable alternative (for example, the GNU gcc compiler)
> and then set the CC variable environment to the path to that compiler, before running the setup.py script. The following
> command can be used to do so (you can find the path to your compiler using the `which` Linux command)
> ``` export CC=<path/to/your/gcc>; python setup.py install ```

> :warning: During this setup, you may encounter errors, as sometimes the Python package dependencies are installed in 
> an order that does not seem to work out. Frequently, installing the package triggering the error by itself before reattempting
> the command that failed seemed to solve these issues (often observed with `pybind11` and `pyscf`).

If you do not wish to use the code in `qSDK` or `agnostic_simulator` as python modules but want to manipulate and 
access this code directly, you can also remove them from your environment (through the `uninstall` command of pip or conda), 
and simply add the path to the root directory of `qSDK` to the beginning of your `PYTHONPATH` environment variable.

### 2.c Using Docker

```
docker build . -t qsdk_docker_image"
```
You can then use `docker run` to instantiate a container, and `docker exec` to start an interactive bash session.

### 3. Optional dependencies

The `backendbuddy` submodule enables users to target various backends. In particular, it supports quantum circuit 
simulators such as `qulacs`, `qiskit`, `cirq` or `qdk`. We leave it to the user to install the package of their choice.
Depending on your OS and environment, some of these packages may be more challenging to install. In particular, we
recommend `qulacs` for its performance and `qiskit` for providing the utilities for the QASM format.

Most packages can be installed through pip in a straightforward way:
```
pip install qulacs
pip install qiskit
pip install cirq
```

For installing Microsoft's `qdk` or any issue regarding the above packages, please check their respective documentation.


## Tests

Unit tests can be found in the `tests` folders, located in the various toolboxes they are related to. To automatically
find and run all tests (assuming you are in the `qsdk` subfolder that contains the code of the package):
```
python -m unittest
```

## Environment variables

This section allows for keeping track of useful environment variables the user could set, in order to adjust the behavior
of their code.
