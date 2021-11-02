#!/bin/bash

# Make sure you build qsdk and all relevant dependencies before attempting to generate documentations
# or mock the desired packages in docs/source/conf.py
cd ../docs || cd docs
pip install sphinx sphinx_rtd_theme nbsphinx
sudo apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended pandoc dvipng
make clean; make html
