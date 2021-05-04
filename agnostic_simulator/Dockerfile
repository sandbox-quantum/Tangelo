FROM fedora:30

RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git (required for qulacs)
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Python modules for documentation, Jupyter notebook support, visualization
# and some computational packages
RUN pip3 install ipython jupyter numpy scipy pybind11 requests pandas \
    setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

RUN pip3 install bitarray

# For test purposes, to define a qubit operator. Interface identical to mol2qb qubit operator
RUN pip3 install openfermion

################### Backends: local simulators ######################
RUN pip3 install qiskit-terra qiskit-aer qiskit-ignis qiskit-ibmq-provider
RUN pip3 install qulacs
RUN pip3 install projectq
RUN pip3 install amazon-braket-sdk

################### Microsoft QDK environment ###################
# Install QDK
WORKDIR /tmp/
RUN dnf clean all
RUN rpm --import https://packages.microsoft.com/keys/microsoft.asc
RUN wget -q -O /etc/yum.repos.d/microsoft-prod.repo https://packages.microsoft.com/config/fedora/30/prod.repo
RUN dnf install -y dotnet-sdk-3.1

# Install the IQSharp backend
RUN dotnet tool install -g Microsoft.Quantum.IQSharp
RUN /root/.dotnet/tools/dotnet-iqsharp install --user

# Install qsharp python package
RUN pip3 install qsharp

########################### Finalize #############################

# Set the python path to find agnostic_simulator
ENV PYTHONPATH=$PYTHONPATH:/root/agnostic_simulator

WORKDIR /root/
