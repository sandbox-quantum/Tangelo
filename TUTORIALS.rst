Tutorials
=========

Setting-up a local environment to run jupyter notebooks can be a cumbersome step for uninitiated python users.

`Google Colab <https://colab.research.google.com/>`_ provides a turnkey coding environment for researchers.

Users can therefore read, execute and collaborate on a code base without leaving the comfort of an internet browser.

The code is executed in the cloud, thus making the code ready-to-use on every platform that has a cloud connection.


Compatible notebooks in Google Colab
------------------------------------

The compatible notebooks are listed in the table below.

| Notebook | GColab link |
|----------|-------------|
| VQE | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/vqe.ipynb |
| Custom VQE | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/vqe_custom_ansatz_hamiltonian.ipynb |
| ADAPT-VQE | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/adapt.ipynb |
| DMET | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/dmet.ipynb |
| ONIOM | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/oniom.ipynb |
| Classical Shadows | https://colab.research.google.com/github/AlexandreF-1qbit/Tangelo/blob/gcolab/examples/classical_shadows.ipynb |

However, adapting a notebook to Google Colab is an easy task.

Installing a package is something possible, and is explained at this `link <https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb>`_.

For our use case, installing Tangelo can be done with this code cell.

.. code-block:: python

  try:
    import tangelo
  except ModuleNotFoundError:
    !pip install git+https://github.com/goodchemistryco/Tangelo.git@develop --quiet
