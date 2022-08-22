FROM fedora:30

# Fundamentals
# ============
RUN dnf -y update
RUN dnf -y install wget libgomp openblas-devel pandoc
RUN dnf clean all

# Python, C/C++ compilers, git
# ============================
RUN dnf -y install gcc redhat-rpm-config gcc-c++ python3-devel make cmake git

# Set up a virtual environment, set all calls to pip3 and python3 to use it
# =========================================================================
RUN pip3 install virtualenv
ENV VIRTUAL_ENV=/root/env
RUN virtualenv -p python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN ls -al /root/.bashrc
RUN more /root/.bashrc
RUN echo "export PATH=$PATH" >> /root/.bashrc

# Python packages for documentation, Jupyter notebook support and visualization
# =============================================================================
RUN pip3 install --upgrade pip
RUN pip3 install ipython jupyter setuptools wheel sphinx py3Dmol sphinx_rtd_theme nbsphinx scikit-build

# Install Tangelo and its immediate dependencies (pyscf, openfermion, ...)
# ========================================================================

# > Option 1: install from pypi
RUN pip3 install tangelo-gc

# > Option 2: install from locally mounted Tangelo, in the docker container
# ENV PYTHONPATH=/root/tangelo:$PYTHONPATH
# WORKDIR /root/
# COPY . /root
# RUN python3 -m pip install .

# OPTIONAL: common dependencies (quantum circuit simulator and quantum cloud services)
# ====================================================================================
 RUN pip3 install cirq amazon-braket-sdk qiskit qulacs projectq
