#!/bin/bash

cd ../docs || cd docs
pip install sphinx sphinx_rtd_theme nbsphinx
sudo apt-get install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended pandoc dvipng
make clean; make html
