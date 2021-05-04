#!/bin/bash

# Run unit tests
more /root/.bashrc
source /root/.bashrc
cd /root/qsdk/qsdk
python3 -m unittest discover
