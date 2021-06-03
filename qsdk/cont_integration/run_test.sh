#!/bin/bash

# qsdk and agn sim to be found during tests
export PYTHONPATH=$PYTHONPATH:/root/agnostic_simulator:/root/qsdk

# Tests agn sim
cd /root/agnostic_simulator/tests/
#/root/env/bin/python3 -m unittest

# Tests qsdk
cd /root/qsdk/
#/root/env/bin/python3 -m unittest

# Tests jupyter
cd /root/qsdk/examples
sh run_notebooks.sh
cd /root/agnostic_simulator/examples
#sh run_notebooks.sh
