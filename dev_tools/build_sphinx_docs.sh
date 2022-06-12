#!/bin/bash

# Make sure you build tangelo and all relevant dependencies before attempting to generate documentations
# or mock the desired packages in docs/source/conf.py
pip install sphinx sphinx_rtd_theme nbsphinx

# Support for LaTeX
sudo apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended pandoc dvipng

# Build html documentation in ../docs/source/html
# Make sure Tangelo is installed beforehand
cd ../docs || cd docs
sphinx-apidoc -o ./source ../tangelo
make clean; make html
cd -
