qSDK overview
=============

Welcome !
This open-source quantum SDK provides tools developed for exploring quantum chemistry simulation end-to-end workflows on 
both gate-model quantum computers and quantum circuit emulators.

It was designed to support the development of quantum algorithms and workflows running by providing building-blocks from various toolboxes.
It attempts to cover all steps of the process, such as quantum execution, pre- and post-processing techniques, including problem decomposition.
It provides users with features such as resource estimation, access to various compute backends (noiseless and noisy simulators, 
quantum devices) and some classical solvers in order to keep track of both accuracy and resource requirements of our workflows,
and facilitate the design of successful hardware experiments.

Install
-------

This package requires a Python 3 environment.

We recommend:


* using `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ in order to set up your environment safely and cleanly
* installing the "dev" version of Python3 if you encounter missing header errors, such as ``python.h file not found``.
* having good C/C++ compilers and BLAS library to ensure the quantum circuit simulators you choose to install have good performance.

Using pip
^^^^^^^^^

TODO: once this package is available on pypi, give the command.

From source, using setuptools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package can be installed by first cloning this repository with ``git clone``\ , and typing the following command in the
root directory:

.. code-block::

   python -m pip install .

If the installation of a dependency fails and the reason is not obvious, we suggest installing that dependency
separately with ``pip``\ , before trying again.

If you would like to modify or develop code in ``qSDK``\ , you can add the path to this folder to your ``PYTHONPATH`` 
environment variable instead of installing it with pip: 

.. code-block::

   export PYTHONPATH=<path_to_this_folder>:$PYTHONPATH

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

qSDK enables users to target various backends. In particular, it integrates quantum circuit  simulators such as 
``qulacs``\ , ``qiskit``\ , ``cirq`` or ``qdk``. We leave it to you to install the packages of your choice.
Backends such as ``qulacs`` and ``cirq`` show good overall performance. Most packages can be installed through pip in a straightforward way:

.. code-block::

   pip install qulacs
   pip install qiskit
   pip install cirq
   ...

Depending on your OS and environment, some of these packages may be more challenging to install. For installing Microsoft's QDK 
or any issue regarding the above packages, please check their respective documentation.

Optional: environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some environment variables can impact performance (ex: using GPU for quantum circuit simulation, or changing
the number of CPU threads used) or are used to connect to web services providing access to some compute backends.

See the list of relevant environment variables and their use in ``env_var.sh``. In order for these variables to be set to
the desired values in your environment, you can run this shell script in Linux with the following command line:
``source env_var.sh`` (you may need to set execution permissions with ``chmod +x set_env_var.sh`` first), or you can set
them in whatever way your OS supports it, or even inside your python script using the ``os`` package.

Docs
----

TODO: insert sentence and link to sphinx documentation when its online.

Tutorials
---------

Please check the ``examples`` folder jupyter notebook tutorials and other examples.
TODO: recommend here the one we are doing about the DMET paper once its ready and merged

Tests
-----

Unit tests can be found in the ``tests`` folders, located in the various toolboxes they are related to. To automatically
find and run all tests (assuming you are in the ``qsdk`` subfolder that contains the code of the package):

.. code-block::

   python -m unittest

Citations
---------

If you use qSDK in your research, please cite

[TODO: this is a placeholder for our qSDK paper, to be written and put on arxiv in October]

Copyright 1QBit 2021. This software is released under the Apache Software License version 2.0.
