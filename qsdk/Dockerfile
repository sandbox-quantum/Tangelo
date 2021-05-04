FROM fedora:30

RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Set up a virtual environment, set all calls to pip3 and python3 to use it
RUN pip3 install virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN virtualenv -p python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Python modules for documentation, Jupyter notebook support, visualization
# and some computational packages
RUN pip3 install --upgrade pip
RUN pip3 install ipython jupyter numpy scipy pyscf pybind11 requests pandas \
    setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

# Copy and set root directory, enable test script to be run
ENV PYTHONPATH=$PYTHONPATH:/root/qsdk/qsdk:/root/agnostic_simulator/agnostic_simulator
WORKDIR /root/
COPY . /root
RUN chmod -R 777 /root/cont_integration/run_test.sh

# Install agnostic simulator
RUN python3 /root/agnostic_simulator/setup.py install

# Install qSDK
RUN python3 /root/qsdk/setup.py install

# Run qSDK tests
RUN cd /root/qsdk/qsdk
RUN python3 -m unittest
