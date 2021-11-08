import setuptools
import sys
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


with open('README.rst', 'r') as f:
    long_description = f.read()

__title__ = "1QBit's quantum SDK for quantum chemistry"
__copyright__ = "1QBit Inc"
__version__ = "0.2.0"
__status__ = "beta"
__authors__ = ["Valentin Senicourt, Alexandre Fleury, Ryan Day, James Brown"]

install('wheel')
install('pyscf')
install('git+https://github.com/pyscf/semiempirical')

setuptools.setup(
    name="qSDK",
    version=__version__,
    description="1QBit's quantum SDK for quantum chemistry on quantum computers and simulators",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/quantumsimulation/QEMIST_qSDK",
    packages=setuptools.find_packages(),
    test_suite="qsdk",
    install_requires=['h5py', 'bitarray', 'openfermion', 'openfermionpyscf']
)
