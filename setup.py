import setuptools
import sys
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


with open('README.rst', 'r') as f:
    long_description = f.read()

install('wheel')
install('h5py==3.2.0')
install('pyscf==1.7.6.post1')

setuptools.setup(
    name="tangelo",
    author="The Tangelo developers",
    version="0.2.0",
    description="Open-source quantum SDK developed for exploring quantum chemistry simulation end-to-end workflows on "
                "gate-model quantum computers",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/quantumsimulation/QEMIST_Tangelo",
    packages=setuptools.find_packages(),
    test_suite="tangelo",
    setup_requires=['h5py'],
    install_requires=['h5py', 'bitarray', 'openfermion', 'openfermionpyscf']
)
