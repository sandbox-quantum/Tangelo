#!/bin/bash

main()
{
  # Root folder (can be modified for the purpose of local testing)
   ROOT=/root
   PYTHON=/root/env/bin/python3
#  ROOT=/media/sf_QEMIST_qSDK
#  PYTHON=/home/valentin/Desktop/virtualenvs/qsdk_may2021/bin/python

  # qsdk and agn sim to be found during tests
  export PYTHONPATH=$PYTHONPATH:$ROOT/agnostic_simulator:$ROOT/qsdk

  # Tests agn sim
  cd $ROOT/agnostic_simulator/tests/
  $PYTHON -m unittest

  # Notebooks agn sim
  TARGET=$ROOT/agnostic_simulator/examples
  run_notebooks_as_unittests $TARGET
  cd -

  # Tests qsdk
  cd $ROOT/qsdk/
  $PYTHON -m unittest

  # Notebooks qsdk
  TARGET=$ROOT/qsdk/examples
  run_notebooks_as_unittests $TARGET
  cd -
}

run_notebooks_as_unittests()
{
  TARGET_PATH=$1
  cd $TARGET_PATH

  # Turn notebooks into scripts, run them with unittest
  NOTEBOOKS=$(ls *.ipynb)
  echo $NOTEBOOKS
  for notebook in $NOTEBOOKS; do
    jupyter nbconvert --to script $notebook
  done

  chmod 777 *.py
  $PYTHON -m unittest
}

#pytest --nbmake "dmet.ipynb"
#bash run_notebooks.sh
#cd /root/agnostic_simulator/examples
#bash run_notebooks.sh

main "$@"
