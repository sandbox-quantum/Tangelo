Tangelo overview
================

|maintainer|
|licence|
|systems|
|dev_branch|
|build|

.. |maintainer| image:: https://img.shields.io/badge/Maintainer-GoodChemistry-blue
   :target: https://goodchemistry.com
.. |licence| image:: https://img.shields.io/badge/License-Apache_2.0-green
   :target: https://github.com/goodchemistryco/Tangelo/blob/main/LICENSE
.. |systems| image:: https://img.shields.io/badge/OS-Linux%20MacOS%20Windows-7373e3
.. |dev_branch| image:: https://img.shields.io/badge/DevBranch-develop-yellow
.. |build| image:: https://github.com/goodchemistryco/Tangelo/actions/workflows/continuous_integration.yml/badge.svg
   :target: https://github.com/goodchemistryco/Tangelo/actions/workflows/continuous_integration.yml

Welcome !

Tangelo is an open-source and free Python package developed by `Good Chemistry Company <https://goodchemistry.com>`_, focusing on the development of end-to-end material simulation workflows on quantum computers.

Easy to pick up, it facilitates the exploration of end-to-end workflows leveraging reusable building blocks or your own custom code, while keeping track of quantum resource requirements, such as number of qubits, gates or measurements.
Through problem decomposition techniques, users can scale up beyond toy models and study the impact of quantum computing on more industrially-relevant use cases. Tangelo is backend-agnostic and compatible with many existing open-source frameworks (Cirq, Qulacs, Qiskit, Braket ...). It is our wish to develop a community around Tangelo, collaborate, and together leverage the best of what the field has to offer.


Install
-------

This package requires a Python 3 environment. We recommend:

* using `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ in order to set up your environment safely and cleanly
* installing the "dev" version of Python3 if you encounter missing header errors, such as ``python.h file not found``.
* having good C/C++ compilers and BLAS libraries to ensure good overall performance of computation-intensive code.

Quick note for Windows users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our installation instructions will work on Linux and MacOS systems. If you are using Windows, we recommend
you install the `Windows Linux Subsystem <https://docs.microsoft.com/en-us/windows/wsl/install>`_, which allows you
to run Ubuntu as an application. Once it has been installed, you can type ``explorer.exe`` in your Ubuntu terminal to
drag and drop files between your Windows and Linux environment.

Here are a few essentials to install inside a brand new Ubuntu environment, before trying to install Tangelo:

.. code-block::

   sudo apt update && sudo apt upgrade
   sudo apt-get install python3-dev
   sudo apt-get install python3-venv
   sudo apt-get install cmake unzip

Using pip
^^^^^^^^^

TODO: once this package is available on pypi, give the command.

From source, using setuptools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package can be installed locally by copying the contents of this repository to any machine.
Type the following command in the root directory:

.. code-block::

   python -m pip install .

If the installation of a dependency fails and the reason is not obvious, we suggest installing that dependency
separately with ``pip``\ , before trying again.


Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

Tangelo enables users to target various backends. In particular, it integrates quantum circuit simulators such as 
``qulacs``\ , ``qiskit``\ , ``cirq`` or ``qdk``. We leave it to you to install the packages of your choice.
Most packages can be installed through pip in a straightforward way:

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

The ``examples`` folder of this repository contains various Jupyter notebook tutorials, and other examples.
We wrote a number of them, but nothing prevents users from contributing more notebook content !
You can visualize a number of pre-run notebooks directly on Github or in our Sphinx documentation. If you'd like to be able to run
them locally, we suggest you use `Jupyter notebooks inside a virtual environment <https://janakiev.com/blog/jupyter-virtual-envs/>`_.

- Install Jupyter in your environment:
.. code-block::

   pip install jupyter

- To make sure the notebooks allow you to set the kernel corresponding to your virtual environment:
.. code-block::

   pip install --user ipykernel
   python -m ipykernel install --user --name=myenv

Tests
-----

Unit tests can be found in the ``tests`` folders, located in the various toolboxes they are related to. To automatically
find and run all tests (assuming you are in the ``tangelo`` subfolder that contains the code of the package):

.. code-block::

   python -m unittest


Contributions
-------------

Please have a look at the `contributions <./CONTRIBUTIONS.rst>`_ file.

Citations
---------

If you use Tangelo in your research, please cite:

[TODO: this is a placeholder for our Tangelo paper, to be written and put on arxiv in October]

© Good Chemistry Company 2021. This software is released under the Apache Software License version 2.0.
