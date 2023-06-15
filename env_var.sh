#!/bin/bash


# 1. Performance
# ---------------------------------------------------------------

# Multithreading: OpenMP threads used in libraries such as quantum simulation backends
export OMP_NUM_THREADS=

# TODO: replace with a flag that just says USE_GPU. Everything GPU-enabled and supported will be accelerated.
# Quantum simulator: use GPUs, if qulacs-gpu has been installed (values: 0 or 1)
export QULACS_USE_GPU=


# 2. QPU Connections
# ---------------------------------------------------------------

# IBM Q services: Qiskit & Qiskit runtime need to be installed, you also
# need a IBM Q account set up.
export IBM_TOKEN=

# IONQ QPU services: your personal APIKEY to use IonQ REST services
export IONQ_APIKEY=

# Honeywell QPU services: your personal login info to use Honeywell REST services
export HONEYWELL_EMAIL=
export HONEYWELL_PASSWORD=

# Amazon Braket services (set variables here or use ~/.aws/credentials)
# You need to have an account and the Amazon CLI installed. Check out their documentation.
export AWS_REGION=
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_SESSION_TOKEN=
