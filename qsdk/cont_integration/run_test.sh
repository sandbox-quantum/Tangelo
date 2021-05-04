#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/root/agnostic_simulator:/root/qsdk

cd /root/agnostic_simulator/tests/
/root/env/bin/python3 -m unittest

cd /root/qsdk/
/root/env/bin/python3 -m unittest

/root/env/bin/pip3 freeze
