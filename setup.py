import setuptools

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
    long_description=long_description,
    url="https://github.com/goodchemistryco/Tangelo",
    packages=setuptools.find_packages(),
    test_suite="tangelo",
    install_requires=['h5py', 'bitarray', 'openfermion'],
    extras_require={
        'pyscf': ['pyscf', 'pyscf-semiempirical @ git+https://github.com/pyscf/semiempirical@v0.1.0'], # pyscf-semiempirical PyPI sdist is missing C extension files
    }
)
