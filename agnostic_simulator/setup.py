import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

__title__ = "Hardware Agnostic Simulator of Quantum Circuits"
__copyright__ = "1QBit Inc"
__version__ = "0.1.0"
__status__ = "beta"
__authors__ = ["Valentin Senicourt"]
__credits__ = ["Valentin Senicourt"]


setuptools.setup(
    name="agnostic_simulator",
    version=__version__,
    description="Hardware-agnostic simulator package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1QB-Information-Technologies/agnostic_simulator",
    packages=setuptools.find_packages(),
    test_suite="agnostic_simulator",
    setup_requires=['pybind11'],
    install_requires=['pybind11', 'numpy', 'scipy', 'bitarray', 'requests', 'pandas', 'openfermion']
)
