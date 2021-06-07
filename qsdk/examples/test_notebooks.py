import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError 

ep =ExecutePreprocessor(timeout=600, kernel_name='python3')

#Run DMET notebook
with open('dmet.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
try:
     out=ep.preprocess(nb)
except CellExecutionError:
     out=None
     msg='Error executing the notebook "%s".\n\n' % 'dmet.ipynb'
     msg+= 'See notebook "%s" for the traceback.\n\n' % 'dmet.ipynb'
     print(msg)
     raise

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
