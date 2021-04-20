FROM fedora:30

RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Python modules for documentation, Jupyter notebook support, visualization
# and some computational packages
RUN pip3 install ipython jupyter numpy scipy pyscf pybind11 requests pandas \
    setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

# TODO: install agnostic simulator and dependencies using setuptools


########################### Finalize #############################
# TODO: install qsdk dependencies. Do we install the qsdk package or simply add it to the path?

ENV PYTHONPATH=$PYTHONPATH:/root/qsdk

WORKDIR /root/
