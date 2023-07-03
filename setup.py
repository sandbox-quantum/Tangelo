import setuptools
import sys
import subprocess
import os

with open("tangelo/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open('README.rst', 'r') as f:
    long_description = f.read()

description = "Maintained by Good Chemistry Company, focusing on the development of end-to-end materials simulation workflows on quantum computers."

setuptools.setup(
    name="tangelo-gc",
    author="The Tangelo developers",
    version=version,
    description=description,
    long_description=description,
    url="https://github.com/goodchemistryco/Tangelo",
    packages=setuptools.find_packages(),
    test_suite="tangelo",
    install_requires=['h5py', 'bitarray', 'openfermion', 'pyscf', 'pyscf-semiempirical']
)
