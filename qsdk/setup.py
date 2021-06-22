import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

__title__ = "1QBit's quantum SDK for quantum chemistry"
__copyright__ = "1QBit Inc"
__version__ = "0.1.0"
__status__ = "beta"
__authors__ = ["Valentin Senicourt, Alexandre Fleury, Ryan Day"]


setuptools.setup(
    name="qSDK",
    version=__version__,
    description="1QBit's quantum SDK for quantum chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1QB-Information-Technologies/QEMIST_qSDK",
    packages=setuptools.find_packages(),
    test_suite="qsdk",
    setup_requires=['agnostic_simulator'],
    install_requires=['h5py==2.9.0', 'pyscf', 'openfermion', 'openfermionpyscf']
)
