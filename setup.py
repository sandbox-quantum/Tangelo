import setuptools
import sys
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

version = '0.3.0'

with open('README.rst', 'r') as f:
    long_description = f.read()

install('wheel')
install('pyscf')
install('git+https://github.com/pyscf/semiempirical')

description = "Maintained by Good Chemistry Company, focusing on the development of end-to-end materials simulation workflows on quantum computers."

setuptools.setup(
    name="tangelo-gc",
    author="The Tangelo developers",
    version=version,
    description=description,
    long_description=description,
    #long_description_content_type=description,
    url="https://github.com/goodchemistryco/Tangelo",
    packages=setuptools.find_packages(),
    test_suite="tangelo",
    setup_requires=['h5py'],
    install_requires=['h5py', 'bitarray', 'openfermion', 'openfermionpyscf']
)
