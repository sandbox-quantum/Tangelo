import setuptools

with open("tangelo/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

# Ignoring this currently, the content of the README is very complex.
with open('README.rst', 'r') as f:
    long_description = f.read()

description = "Tangelo is an open-source Python package maintained by Good Chemistry Company, focusing on the development of quantum chemistry simulation workflows on quantum computers. It was developed as an engine to accelerate research, and leverages other popular frameworks to harness the innovation in our field."

setuptools.setup(
    name="tangelo-gc",
    author="The Tangelo developers",
    version=version,
    description=description,
    long_description=description,
    url="https://github.com/goodchemistryco/Tangelo",
    packages=setuptools.find_packages(),
    test_suite="tangelo",
    install_requires=['h5py', 'bitarray', 'openfermion'],
    extras_require={
        'pyscf': ['pyscf'] #'pyscf-semiempirical @ git+https://github.com/pyscf/semiempirical@v0.1.0'], # pyscf-semiempirical PyPI sdist is missing C extension files
    }
)
