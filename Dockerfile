FROM fedora:30

RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Python modules for documentation, Jupyter notebook support, visualization
# and some computational packages
RUN python -m pip install -U pip
RUN pip3 install ipython jupyter numpy scipy pyscf pybind11 requests pandas \
    setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

#RUN pip3 install urllib3
#RUN cd /usr/local/lib/python3.*/site-packages/urllib3-*.dist-info && pwd && ls -l
#&& cp metadata.json METADATA && ls METADATA

ENV PYTHONPATH=$PYTHONPATH:/root/qsdk

WORKDIR /root/
COPY . /root
RUN chmod -R 777 /root/cont_integration/run_test.sh

# Install agnostic simulator
RUN git submodule init && git submodule update
RUN cd /root/agnostic_simulator && python3 setup.py install && cd /root/

# Install qSDK
RUN python3 setup.py install
