#!/bin/bash

# Run unit tests
more /root/.bashrc
source /root/.bashrc
ls -l /root/env/bin
cd /root/qsdk/qsdk
python3 -m unittest discover
