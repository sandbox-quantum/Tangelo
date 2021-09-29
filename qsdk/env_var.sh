#!/bin/bash

# To set the number of OpenMP threads used by the package (may have a strong impact on performance)
export OMP_NUM_THREADS=

# To use GPUs, if qulacs-gpu has been installed (values: 0 or 1)
export QULACS_USE_GPU=

# IONQ QPU services: your personal APIKEY to use IonQ REST services
export IONQ_APIKEY=

# Honeywell QPU services: your personal login info to use Honeywell REST services
export HONEYWELL_EMAIL=
export HONEYWELL_PASSWORD=
