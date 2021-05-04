# Agnostic_simulator

Manage quantum information processing tasks in an agnostic manner, allowing users to target
both classical and quantum backends, for noisy or noiseless quantum circuit simulation, through a common interface.

This package aims at providing users with tools to easily generate and simulate quantum circuits on different backends,
from classical simulators (Qulacs, Qiskit, QDK, ...) to actual quantum processors developed by hardware partners
(IonQ, Honeywell...). These backends have different features and performance, and may be appealing for different
reasons, whether they are to be used for stand-alone experiments destined to run on a QPU, or to be used as the compute
backbone of applications relying on the simulation of quantum circuits.

On top of that, this package can host any innovation aiming at improving the resource requirements, reducing the number
of measurements, investigating noise-correction techniques or other forms of post-processing related to quantum circuit simulation.

## Contents of the repository

Details the organization of this repository and the contents of each folder

- **agnostic_simulator** :
Source code of this package

- **tests** :
Tools and script for continuous integration (versioning, automated testing and updating documentation)

- **docs** :
Source code documentation and user documentation (readthedocs)

- **examples** :
Examples and tutorials to learn how to use the different functionalities of the library

- **notes** :
Additional notes on the project

## Install

You need to have Python3 installed (Python 2 is not supported anymore as of Jan 1st 2020, and has been officially replaced by Python 3)
in your environment. We recommend you use Python virtual environments (virtualenv) in order to set up your environment safely and cleanly, 
in the following. Brief overviews of virtualenv are available online, such as: https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv

We heavily recommend you have a good C/C++ compiler supporting modern C++ and OpenMP multithreading installed, 
such as the GNU ones (gcc, g++), as well as your own BLAS libraries. Your OS's package manager should provide a 
straightforward way to install these. This will have a big impact on the execution time of your code.

> :warning: You may encounter "python.h file not found" errors: ensure that you have the "dev" version of python3
> installed in your environment. Installation instructions may differ depending on your operating system, but are generally
> a one-liner using your distribution's package manager.

### Using setup.py

This package can be installed into your active Python environment by using the setuptools. 
In a terminal open in the root directory of this project, type the following to install the package and its dependencies
`python setup.py install`

Note: This package relies on `openfermion` 's `QubitOperator` class in order to represent a qubit operator, which is
used to facilitate the computation of expectation values.

Note: Installation does not include the Microsoft QDK and qsharp Python package. The QDK code or tests will therefore
not work as expected if you are choose this installation method. To install QDK and the qsharp Python package, please have a look at
https://docs.microsoft.com/en-us/quantum/install-guide/ , or the contents of the Dockerfile we provide (see section below).

> :warning: If you wish to use qulacs with a GPU, you first need a working CUDA installation. I suggest you then install
> `qulacs-gpu` with `pip install qulacs-gpu` and then install `agnostic_simulator` to get the rest of the dependencies. 
> To use the GPU, make sure that you have set the corresponding environment variable: `export QULACS_USE_GPU=1`.

### Using Docker

This directory contains a Dockerfile building a Fedora image with all dependencies installed, including QDK and Openfermion,
able to run all the tests and examples of the package.
To first build the docker image: 
`docker build . -t agnostic_simulator`
You can then use `docker run` to instantiate a container.

## Environment variables

See the list of relevant environment variables and their use in env_var.sh
