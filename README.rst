[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/nasa/hybridq.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nasa/hybridq/context:python)

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

Tangelo overview
================

|maintainer|
|licence|
|build|

.. |maintainer| image:: https://img.shields.io/badge/Maintainer-GoodChemistry-blue
.. |licence| image:: https://img.shields.io/badge/License-Apache_2.0-green
   :target: https://github.com/quantumsimulation/QEMIST_qSDK/blob/main/README.rst
.. |build| image:: https://github.com/quantumsimulation/QEMIST_qSDK/actions/workflows/github_actions_automated_testing.yml/badge.svg
.. |python| image:: 

Welcome !

Tangelo is an open-source python package developed by Good Chemistry Company, focused on the development of end-to-end material simulation workflows on quantum computers. Its modular design and ease-of-use enables users to easily assemble custom workflows, tinker and define their own building blocks, while keeping track of quantum resource requirements, such as number of qubits, gates or measurements. Through problem decomposition techniques, users can scale up beyond toy models and study the impact of quantum computing on more industrially-relevant use cases. Tangelo is backend-agnostic and compatible with many existing open-source frameworks, making the integration of third-party tools such as state-of-the-art simulators, circuit compilers or quantum cloud services straightforward. It is our wish to develop a community around Tangelo, collaborate, and together leverage the best of what the field has to offer.


Install
-------

This package requires a Python 3 environment. We recommend:

* using `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ in order to set up your environment safely and cleanly
* installing the "dev" version of Python3 if you encounter missing header errors, such as ``python.h file not found``.
* having good C/C++ compilers and BLAS libraries to ensure good overall performance of computation-intensive code.

Using pip
^^^^^^^^^

TODO: once this package is available on pypi, give the command.

From source, using setuptools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package can be installed by downloading or cloning the contents of this repository, and typing the following command in the
root directory:

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

Please check the ``examples`` folder jupyter notebook tutorials and other examples.


Tests
-----

Unit tests can be found in the ``tests`` folders, located in the various toolboxes they are related to. To automatically
find and run all tests (assuming you are in the ``tangelo`` subfolder that contains the code of the package):

.. code-block::

   python -m unittest

Citations
---------

If you use Tangelo in your research, please cite

[TODO: this is a placeholder for our Tangelo paper, to be written and put on arxiv in October]

Copyright Good Chemistry Company 2021. This software is released under the Apache Software License version 2.0.
