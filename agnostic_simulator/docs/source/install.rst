
*******************
Install guide
*******************

.. contents:: Table of Contents


Through setup.py
================

This package can be installed into your active Python environment by using the setuptools.
In a terminal open in the root directory of this project, type the following to install the package and its dependencies:
``python setup.py install``

Note: This package does not provide a qubit operator class, but offers functionalities - such as computing the
expectation value of a qubit operator - that expect as input an operator whose terms are represented similarly to
the ``QubitOperator`` class in Openfermion (https://github.com/quantumlib/OpenFermion/blob/master/src/openfermion/ops/_symbolic_operator.py).
Some examples are available in the `examples` and `tests` sections. You can simply install Openfermion through pip:
```pip install openfermion``` (https://github.com/quantumlib/OpenFermion).

Note: This does not include the Microsoft QDK and qsharp Python package. The QDK code or tests will therefore
not work as expected if you are choose this installation method. To install QDK and the qsharp Python package, please have a look at
https://docs.microsoft.com/en-us/quantum/install-guide/ , or the contents of the Dockerfile we provide (see section below).

Note: If you wish to use qulacs with a GPU, you need a working CUDA installation. I suggest you first install
`qulacs-gpu` with ```pip install qulacs-gpu``` and then install `agnostic_simulator` to get the rest of the dependencies.
To use the GPU, make sure that you have set the corresponding environment variable: ```export QULACS_USE_GPU=1```.

Through Docker
==============

This directory contains a Dockerfile building a Fedora image with all dependencies installed, including QDK and Openfermion,
able to run all the tests and examples of the package.
To first build the docker image:
``docker build . -t agnostic_simulator``
You can then use ``docker run`` to instantiate a container.
