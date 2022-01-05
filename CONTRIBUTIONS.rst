Contributions guidelines
========================

Thank you very much for considering contributing to this project; we’d love to have you on board ! Do not feel intimidated by the guidelines and processes we describe in this document: we are here to assist you and help you take things to the finish line. We do not expect you to be an expert in software development or to get everything right on the first attempt: don’t hesitate to open an issue or a pull request, or simply contact us.

Contributors have various backgounds and experience, from high schoolers to fully fledged quantum scientists or chemists, and there are many ways you can contribute to this project. You can of course open a pull request and extend our codebase, but opening an issue to suggest a new feature, report a bug, improve our documentation or make a Python notebook are just as valuable.

By joining the Tangelo community, you gain the opportunity to contribute to a collaborative project, in order to advance the field of quantum computing and further develop your own skills.

This package is under licence `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_.


Pull request and code review process
------------------------------------

All submissions to the Github repository are subject to review by qualified project members. This is done through `Github’s Pull Request process <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_.

We recommend you fork the `main Tangelo repo <https://github.com/quantumsimulation/QEMIST_qSDK>`_, create and work on your development branch on this fork and then create a Pull Request (PR) to the main tangelo repo.


**1. Set up your fork**

Go to the `main Tangelo repo <https://github.com/quantumsimulation/QEMIST_qSDK>`_ and click the Fork button in the upper right corner of the screen.
This creates a new Github repo ``https://github.com/USERNAME/tangelo`` where ``USERNAME`` is your Github ID.

In your terminal, clone the repo on your local machine, and move into the newly created directory:

.. code-block:: shell

  git clone https://github.com/quantumsimulation/QEMIST_qSDK.git
  cd QEMISK_qSDK

From the perspective of your local clone, your fork is called ``origin`` remote. 
Let's synchronize your fork with the main Tangelo repo by adding the latter as the upstream remote. Then update your local main branch:

.. code-block:: shell
git remote add upstream https://github.com/quantumsimulation/QEMIST_qSDK.git

git fetch upstream
git checkout main
git merge upstream/main


**2. Work on your own developments**

Create your development branch, based on the `main` branch (or the current development branch listed on the `DevBranch badge <./README.rst>`_)

.. code-block:: shell

  git checkout main -b your_branch_name

where ``your_branch_name`` is the name of your own development branch, preferably related to what you will be working on.
Let's assume you've made some changes and committed them with ``git commit``, and that you'd like to push them to your fork (which is referred to as "origin"):

.. code-block:: shell

  git push origin new_branch_name


**3. The Pull Request (PR)**

Now when you go to https://github.com/quantumsimulation/QEMIST_qSDK, you should be able to create a pull request from the branch on your fork to a branch on the main Tangelo repo.
Give your pull request a name and briefly describe what the purpose is and include a reference to the associated issue if there's one.
Several Tangelo users will receive a notification, and will review your code and leave comments in the PR. You can reply to these comments, or simply apply the recommended changes locally, and then commit and push them like above: it automatically updates your PR.
If there are conflicts, you can solve them locally and push, or directly through Github.

Getting your code reviewed can feel intimidating, but remember it's just part of a standard process: everyone has to go through it (even the main developers) and it is uncommon for PRs to be approved without changes or questions first.
We suggest you have a look at how other files of this project (source code, tests, docs...) are written and follow the same format fom the start: this way most of the work is done. 
We require that you write tests for your code, as well the docstrings for it. Don't worry: we're here to help and there are plenty examples in the repo.
We usually follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/>`_ for our code. If you're using an IDE (Pycharm, etc), it may automatically tell you where your code is not following PEP8 and should be able to automatically reformat your code too.

Every time you open a PR or push more code into an open one, several automated processes are launched and can be monitored in Github. We discuss them in the section below.


## Code Testing Standards

When a pull request is created or updated, various automatic checks will 
run to ensure that the change won't break Cirq and meets our coding standards.

Cirq contains a continuous integration tool to verify testing.  See our
[development page](docs/dev/development.md) on how to run the continuous
integration checks locally.

Please be aware of the following code standards that will be applied to any
new changes.

- **Tests**.
Existing tests must continue to pass (or be updated) when new changes are 
introduced. We use [pytest](https://docs.pytest.org/en/latest/) to run our 
tests.
- **Coverage**.
Code should be covered by tests.
We use [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) to compute 
coverage, and custom tooling to filter down the output to only include new or
changed code. We don't require 100% coverage, but any uncovered code must 
be annotated with `# coverage: ignore`. To ignore coverage of a single line, 
place `# coverage: ignore` at the end of the line. To ignore coverage for 
an entire block, start the block with a `# coverage: ignore` comment on its 
own line.
- **Lint**.
Code should meet common style standards for python and be free of error-prone 
constructs. We use [pylint](https://www.pylint.org/) to check for lint.
To see which lint checks we enforce, see the 
[dev_tools/conf/.pylintrc](dev_tools/conf/.pylintrc) file. When pylint produces
a false positive, it can be squashed with annotations like 
`# pylint: disable=unused-import`.
- **Types**.
Code should have [type annotations](https://www.python.org/dev/peps/pep-0484/).
We use [mypy](http://mypy-lang.org/) to check that type annotations are correct.
When type checking produces a false positive, it can be ignored with 
annotations like `# type: ignore`.

## Request For Comment Process for New Major Features

For larger contributions that will benefit from design reviews, please use the 
[Request for Comment](docs/dev/rfc_process.md) process.

## Developing notebooks 

Please refer to our [notebooks guide](docs/dev/notebooks.md) on how to develop iPython notebooks for documentation.
