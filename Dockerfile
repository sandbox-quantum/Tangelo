FROM fedora:30

RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Set up a virtual environment, set all calls to pip3 and python3 to use it
RUN pip3 install virtualenv
ENV VIRTUAL_ENV=/root/env
RUN virtualenv -p python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN ls -al /root/.bashrc
RUN more /root/.bashrc
RUN echo "export PATH=$PATH" >> /root/.bashrc

# Python packages for documentation, Jupyter notebook support and visualization
RUN pip3 install --upgrade pip
RUN pip3 install h5py==2.9.0 ipython jupyter setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

# Copy and set root directory,
ENV PYTHONPATH=/root/qsdk:$PYTHONPATH
WORKDIR /root/
COPY . /root

# Install qSDK and its immediate dependencies (pyscf, openfermion, ...)
RUN python3 /root/setup.py install

# Install Microsoft QDK qsharp package
WORKDIR /tmp/
RUN dnf clean all
RUN rpm --import https://packages.microsoft.com/keys/microsoft.asc
RUN wget -q -O /etc/yum.repos.d/microsoft-prod.repo https://packages.microsoft.com/config/fedora/30/prod.repo
RUN dnf install -y dotnet-sdk-3.1
RUN dotnet tool install -g Microsoft.Quantum.IQSharp
RUN /root/.dotnet/tools/dotnet-iqsharp install --user
RUN pip3 install qsharp

# Install other simulators
RUN pip install amazon-braket-sdk qiskit qulacs projectq
