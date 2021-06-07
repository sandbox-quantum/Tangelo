import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ep =ExecutePreprocessor(timeout=600, kernel_name='python3')

#Run DMET notebook
with open('dmet.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)

#Run vqe_custom_ansatz.ipynb
with open('vqe_custom_ansatz.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)

#Run vqe_custom_ansatz.ipynb
with open('vqe.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)

#Run vqe_custom_ansatz.ipynb
with open('user_notebooks/adapt.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)
