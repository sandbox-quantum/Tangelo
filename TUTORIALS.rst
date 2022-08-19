Tutorials
=========

* Intro about Tangelo, potential reach and philosophy of our examples.
* Mention jupyter notebook can be saved in Google Drive.
* Put a figure or an explanation on how to open any notebook from github.
* Talk about nbviewer, https://mybinder.org/?
* Better way of presenting links: shields.io badges
* Put a link of this website: https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb

Jupyter notebook python tutorials
---------------------------------

Chemistry, quantum computing and software development is tough stuff. We believe that tutorials in the form of jupyter notebooks are
a great tool to both disseminate knowledge, but also to showcase all the cool stuff the community has done with Tangelo.
Working on notebooks is a great way to contribute to this project, and to show everyone something neat you've worked on:
perhaps something that led to a hardware experiment, or implemented an interesting approach.

If you are new to Jupyter notebooks, check out `this page <https://realpython.com/jupyter-notebook-introduction/>`_.


Quickly deploy a notebook in the cloud
--------------------------------------

Sometimes, you don't want to spend time setting up a local python environment. Maybe it's too cumbersome, or maybe it's
just that your computer's performance is not quite as good as what you'd like. Maybe you simply want to run an existing
notebook and try a few things, right now. Or maybe you intend to run a workshop and you want to avoid any delays
incurred by installation issues experienced by your attendees (the worst). Some cloud providers offer services that can
help with that.

Google colab
^^^^^^^^^^^^

.. |gcolab| image:: https://colab.research.google.com/assets/colab-badge.svg

Google Colab is a rather straightforward way to achieve the above, as explained on this `webpage <https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk>`_.
If you see a |gcolab| badge like this one in a notebook, you can just deploy the notebook on Google Colab by just clicking on it.

Users can read, execute and collaborate through their internet browser. The notebook is hosted and executed on a machine
in the cloud. There are several subscription tiers: the first one is free and may provide you with what you need. The
others can provide you with access to more performant hardware, including GPUs and TPUs, or extra features.

Most of our notebooks are ready to be deployed through Google Colab as-is. A few notebooks require dependencies
that are not publicly available (at the time of writing, QEMIST Cloud is not), or are a bit trickier to install: you may
have to contact us to get access to non-public material.

It is possible that Google Colab is not available in your region of the world. Maybe other cloud providers offer similar
services in your area.

Setting up your environment through an already-deployed notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have access to an already-deployed notebook in a remote environment you have little control on, you can actually make python run shell commands to modify
the environment in which it runs. We use that trick to check right at the start if Tangelo or the other dependencies
for our notebooks are already installed, and install them for you if not. This is what the cell looks like:

.. code-block::

   try:
      import tangelo
   except ModuleNotFoundError:
      !pip install git+https://github.com/goodchemistryco/Tangelo.git@develop --quiet

You can use pip to install python packages, but you can run any other shell command: use other package managers for other
software and libraries, download data from somewhere else...
`These examples <https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb>`_ are not specific to Google Colab.
