#!/bin/bash

# Run unit tests
#more /root/.bashrc
#source /root/.bashrc
#ls -l /root/env/bin
export PYTHONPATH=$PYTHONPATH:/root/agnostic_simulator:/root/qsdk

cd /root/qsdk/
/root/env/bin/python3 -m unittest
