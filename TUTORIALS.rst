Tutorials
=========

Setting-up a local environment to run jupyter notebooks can be a cumbersome step for uninitiated python users.
`Google Colab <https://colab.research.google.com/>`_ provides a turnkey coding environment for researchers.
Users can therefore read, execute and collaborate on a code base without leaving the comfort of an internet browser.
The code is executed in the cloud, thus making the code ready-to-use on every operating system that has internet access.


Compatible notebooks in Google Colab
------------------------------------

The compatible notebooks are listed below:

* `VQE <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/vqe.ipynb>`_
* `Custom VQE <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/vqe_custom_ansatz_hamiltonian.ipynb>`_
* `ADAPT-VQE <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/adapt.ipynb>`_
* `DMET <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/dmet.ipynb>`_
* `ONIOM  <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/oniom.ipynb>`_
* `MIFNO <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/mifno.ipynb>`_
* `Classical Shadows <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/classical_shadows.ipynb>`_
* `linq/1.the_basics <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/linq/1.the_basics.ipynb>`_
* `linq/3.noisy_simulation <https://colab.research.google.com/github/goodchemistryco/Tangelo/blob/develop/examples/linq/3.noisy_simulation.ipynb>`_

However, adapting a new notebook for Google Colab is an easy task.
Installing a package is something possible, and is explained at this `link <https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb>`_.
For our use case, installing Tangelo can be done with this code cell.

.. code-block:: python

  try:
    import tangelo
  except ModuleNotFoundError:
    !pip install git+https://github.com/goodchemistryco/Tangelo.git@develop --quiet
