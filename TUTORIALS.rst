Tutorials
=========

Jupyter notebook python tutorials
---------------------------------

Chemistry, quantum computing and software development is tough stuff. We believe that tutorials in the form of jupyter notebooks are
a great tool to both disseminate knowledge, but also to showcase all the cool stuff the community has done with Tangelo.
Working on notebooks is a great way to contribute to this project, and to show everyone something neat you've worked on:
perhaps something that led to a hardware experiment, or implemented an interesting approach.

If you are new to Jupyter notebooks, check out `this page <https://realpython.com/jupyter-notebook-introduction/>`_. You will learn how to deploy a Jupyter server in your environment and use notebooks.

Quickly deploy a notebook in the cloud
--------------------------------------

Google Colab
^^^^^^^^^^^^

.. |gcolab| image:: https://colab.research.google.com/assets/colab-badge.svg

If you have a Google account (gmail etc) just click the |gcolab| badge on the landing page of our `tutorial repository <https://github.com/goodchemistryco/Tangelo-Examples>`_, or inside any notebook you come across. Notebooks will be launched through your web browser and run on a machine in the cloud. 

Click here for more information about `Google Colab <https://research.google.com/colaboratory/faq.html>`_.

It is possible to install python packages with pip from within an active notebook, by using a `!` to signify a bash command, such as below. Sometimes, you may have to restart the notebook kernel for this change to be taken into account.

.. code-block::

   !pip install git+https://github.com/goodchemistryco/Tangelo.git@develop --quiet
