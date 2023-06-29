Contributions guidelines
========================

Thank you very much for considering contributing to this project !

Do not feel intimidated by the guidelines and processes we describe in this document: we are here to assist you and help you take things to the finish line. We do not expect you to be an expert in software development or to get everything right on the first attempt: don’t hesitate to open an issue or a pull request, or simply contact us.

Contributors have various backgounds and experience, from high schoolers to fully fledged quantum scientists or chemists, and there are many ways you can contribute to this project. You can of course open a pull request (PR) and extend our codebase, but opening an issue to suggest a new feature, report a bug, improve our documentation or make a tutorial notebook is just as valuable.

By joining the Tangelo community and sharing your ideas and developments, you are creating an opportunity for us to learn and grow together, and take ideas to the finish line and beyond.

Tangelo is under licence `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_.


Code of conduct
---------------

Tangelo currently does not have its own code of conduct, but values such as respect and inclusiveness are very important to us. The following `covenant <https://www.contributor-covenant.org/version/1/4/code-of-conduct/>`_ is a good reflection of the kind of environment we want to foster. Please have a quick look.


Feature requests, bug reports
-----------------------------

Have a look at the issue tab, and complete the adequate issue template if needed: there's one for feature request, bug reports, and more. If it turns out the issue ticket you wanted to bring up already exists, please consider leaving a thumbs up or participate in the conversation to help us prioritize or move things forward. It's important to know what matters to users, to take our collaborative project in the right direction: all of this is very useful !


Pull request and code review process
------------------------------------

All submissions to the Github repository are subject to review by qualified project members. This is done through `Github’s Pull Request process <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_. We recommend you fork the `main Tangelo repo <https://github.com/quantumsimulation/QEMIST_qSDK>`_, create and work on your development branch on this fork and then create a Pull Request (PR) to the main tangelo repo.


**1. Set up your fork**

Go to the `main Tangelo repo <https://github.com/goodchemistryco/Tangelo>`_ and click the Fork button in the upper right corner of the screen.
This creates a new Github repo ``https://github.com/USERNAME/Tangelo`` where ``USERNAME`` is your Github ID.

In your terminal, clone the repo on your local machine, and move into the newly created directory (replace ``USERNAME`` with your user ID)

.. code-block:: shell

  git clone https://github.com/USERNAME/Tangelo.git
  cd Tangelo

From the perspective of your local clone, your fork is called the ``origin`` remote.
Let's synchronize your fork with the main Tangelo repo by adding the latter as the upstream remote, and then update your local ``develop`` branch:

.. code-block:: shell

  git remote add upstream https://github.com/goodchemistryco/Tangelo.git

  git fetch upstream
  git checkout develop
  git merge upstream/develop

Note: we here suggest the ``develop`` branch, as this is where contributions will be merged. No one should be merging directly to ``main``, unless it is to sync it with ``develop`` once in a while, and just before a new version release.

**2. Work on your own developments**

Create your development branch, based on the ``develop`` branch (or the current development branch listed on the `DevBranch badge <./README.rst>`_, if different)

.. code-block:: shell

  git checkout develop -b your_branch_name

where ``your_branch_name`` is the name of your own development branch, preferably related to what you will be working on.
Let's assume you've made some changes and committed them with ``git commit``, and that you'd like to push them to your fork (which is referred to as "origin"):

.. code-block:: shell

  git push origin new_branch_name


**3. The Pull Request (PR)**

Now when you go to https://github.com/goodchemistryco/Tangelo, you should be able to create a pull request from the branch on your fork to a branch on the main Tangelo repo. Give your pull request a name, verify that the destination branch is ``develop`` (not ``main``), and briefly describe what the purpose is / include a reference to the associated issue if there's one.
Several Tangelo users will receive a notification, and will review your code and leave comments in the PR. You can reply to these comments, or simply apply the recommended changes locally, and then commit and push them like above: it automatically updates your PR.
If there are conflicts, you can solve them locally and push, or directly through Github.

Getting your code reviewed can feel intimidating, but remember it's just part of a standard process: everyone has to go through it (even the main developers) and it is actually uncommon for PRs to be approved without changes or questions first. We suggest you have a look at how other files of this project (source code, tests, docs...) are written, and follow the same format from the start to avoid having to make a lot of changes to your code later on.

We require that you write tests for your code, as well as the docstrings for it. Don't worry: there are plenty examples in the repo.
We usually follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/>`_ for our code. If you're using an IDE (Pycharm, etc), it may automatically highlight the part of your code that is not following PEP8, and should be able to automatically reformat your code too.

Every time you open a PR or push more code into an open one, several automated processes are launched and can be monitored on Github: we need them to be successful. We elaborate on them in the section below.


Continuous integration
----------------------

When a pull request is created or updated, several automated processes are launched. You will find most of them in the "checks" tab of your pull request, and can look into the details. These processes check for a few things:

**Build**

  This step attempts to build and install both Tangelo and its dependencies using your branch. It is necessary for this to succeed in order for most other checks to run.

**Tests**

  New changes should not break existing features: that's why we're running all the existing tests, on top of your new tests. If something fails, it may be a consequence of your changes, and we should find out what's going on. We use `pytest <https://docs.pytest.org/en/latest/>`_ to run our tests.

  You can run tests locally with unittest; just move to the `tangelo` subfolder of the repo, which contains the source code, and type:

  .. code-block:: shell

    python -m unittest

  This will run all the tests found in the subdirectories, using your local environment (which may not exactly be the one used in the automated tests).
  We also have tests that run a few important example notebooks that can execute quickly.

**Linting / code style**

  A way to check that your code complies with our style guidelines, based on PEP8.
  We rely on a tool called pycodestyle. If you want to know exactly what this linting enforces and ignores, you can refer to this `file <./dev_tools/pycodestyle>`_ and `pycodestyle's documentation <https://pycodestyle.pycqa.org/en/latest/intro.html>`_.


Developing notebooks
--------------------

Jupyter notebooks are great ! If you feel like making a notebook to show how to do something cool with Tangelo and educate others, don't hesitate to reach out. It counts as code, so it will go through the standard PR process and will need to meet a few requirements. The developer team has made several notebooks you can look at, for inspiration.
