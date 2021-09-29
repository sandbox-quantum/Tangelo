# QEMIST_qSDK

Welcome !
This quantum SDK provides tools developed for quantum chemistry simulation on both quantum computers and emulators.

The package provides access to some problem decomposition techniques, electronic structure solvers, as well as the
various toolboxes containing the functionalities necessary to pre-built workflows, and maybe your own.
Users can target a variety of compute backends (QPUs and simulators) available on their local machine or 
through cloud services, using the `backendbuddy` module.




## Contents of the repository

This folder contains the following files and subfolders:

- **qsdk** :
Code implementing various functionalities and algorithms to research and perform quantum chemistry experiments on
  gate-model quantum computers or simulators. The various features are exposed in toolboxes, and can be assembled to
  form workflows: electronic structure solvers.

- **docs** :
Source code documentation and user documentation (Sphinx)

- **examples** :
Examples and tutorials to learn how to use the different functionalities of the package
  

## Install

### 1.Prerequisites

You need to have Python3 installed (Python 2 is not supported anymore as of Jan 1st 2020, and has been officially 
replaced by Python 3) in your environment. We recommend you use Python virtual environments (`virtualenv`) in order to 
set up your environment safely and cleanly, in the following. Brief overviews of virtualenv are available online, 
such as: https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv

We heavily recommend you have a good C/C++ compiler supporting modern C++ and OpenMP multithreading installed, 
such as the GNU ones (gcc, g++), as well as your own BLAS libraries. Your OS's package manager should provide a 
straightforward way to install these. These will have a big impact on the execution time of your code.

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

A few tips below that may be relevant to you:

> :warning: During this setup, you may encounter errors, as sometimes the Python package dependencies are installed in 
> an order that does not seem to work out. Frequently, installing the package triggering the error by itself before reattempting
> the command that failed seemed to solve these issues (often observed with `pybind11` and `pyscf`).

If you are a developer and would like to modify and change code in `qSDK`, you can add the `qsdk` folder that contains 
the source code of this package and add it to your `$PYTHONPATH` environment variable instead of installing it with pip. 

### 2.c Using Docker

```
docker build . -t qsdk_docker_image"
```
You can then use `docker run` to instantiate a container, and `docker exec` to start an interactive bash session.

### 3. Optional dependencies

The `backendbuddy` submodule enables users to target various backends. In particular, it supports quantum circuit 
simulators such as `qulacs`, `qiskit`, `cirq` or `qdk`. We leave it to the user to install the package of their choice.
Depending on your OS and environment, some of these packages may be more challenging to install. In particular, we
recommend `qulacs`, `cirq` and `qiskit` (the latter in particular for its noisy simulation performance).

Most packages can be installed through pip in a straightforward way:
```
pip install qulacs
pip install qiskit
pip install cirq
```

For installing Microsoft's `qdk` or any issue regarding the above packages, please check their respective documentation.

# TODO : Verify with James what variable is needed here, in case we did manage to install qulacs on MacOS
> :warning: If you are using MacOS, your C/C++ compiler may be Clang (even though it masquerades as `gcc`), which does 
> not support compilation of OpenMP multithreaded code. As a consequence, you may encounters errors, or see noticeable 
> degradation in performance for some of the dependencies of this package. 
> We recommend you install a suitable alternative (for example, the GNU gcc compilers, using `brew`)
 > and then export the CC variable environment (or whatever variable is used in their installation script) to the path to 
> that compiler, before running the setup.py script.

## Tests

Unit tests can be found in the `tests` folders, located in the various toolboxes they are related to. To automatically
find and run all tests (assuming you are in the `qsdk` subfolder that contains the code of the package):

```
python -m unittest
```

## Environment variables

This section allows for keeping track of useful environment variables the user could set, in order to adjust the behavior
of their code. Some environment variables can impact performance (ex: using GPU for quantum circuit simulation, or changing
the number of CPU threads used) or are used to connect to web services providing access to some compute backends.

See the list of relevant environment variables and their use in `env_var.sh`. In order for these variables to be set to
the desired values in your environment, you can run this shell script in Linux with the following command line:
`source env_var.sh` (you may need to set execution permissions with `chmod +x set_env_var.sh` first), or you can set
them in whatever way your OS supports it, or even inside your python script using the `os` package.
