#!/bin/bash

runipy dmet.ipynb
runipy vqe_custom_ansatz.ipynb
runipy vqe.ipynb
rm tmp*
cd user_notebooks
runipy adapt.ipynb 
rm tmp* 
