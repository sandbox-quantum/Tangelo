import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ep =ExecutePreprocessor(timeout=600, kernel_name='python3')

#Run 1.the_basics.ipynb notebook
with open('1.the_basics.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)

#Run 3.noisy_simulation.ipynb
with open('3.noisy_simulation.ipynb') as f:
     nb=nbformat.read(f,as_version=4)
ep.preprocess(nb)
