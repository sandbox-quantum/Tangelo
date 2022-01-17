# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utililty functions to generate the FermionOperators from Majorana type operators"""

from openfermion.transforms.opconversions.conversions import get_fermion_operator
from openfermion import MajoranaOperator
from numpy import integer

from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule


def majorana_uccsd_generator(mol: SecondQuantizedMolecule=None, n_electrons: int=None, n_sos:int=None):
    """Constructs a list of FermionOperator corresponding to the individual Majorana modes in the UCCSD ansatz

    Args:
        mol (SecondQuantizedMolecule): The molecule to generate the Majorana pool from: Default None
        n_electrons (int): The number of active electrons: Default None
        n_sos (int): The number of active spin orbitals: Default None

    Returns:
        list: The list of FermionOperator for each Majorana operator in a UCCD pool"""

    if mol is not None:
        n_active_electrons = mol.n_active_electrons
        n_active_sos = mol.n_active_sos
    elif isinstance(n_electrons, (int, integer)) and isinstance(n_sos, (int, integer)):
        n_active_electrons = n_electrons
        n_active_sos = n_sos
    else:
        raise ValueError("SecondQuantized mol or ints n_electrons/n_sos must be provided")
    

    term_set = set()
    for i in range(mol.n_active_electrons):
        for j in range(i+1, mol.n_active_electrons):
            for k in range(mol.n_active_electrons, mol.n_active_sos):
                for l in range(k+1, mol.n_active_sos):
                    term_set.add((2*i, 2*j+1, 2*k+1, 2*l+1))
                    term_set.add((2*i+1, 2*j, 2*k+1, 2*l+1))
                    term_set.add((2*i+1, 2*j+1, 2*k, 2*l+1))
                    term_set.add((2*i+1, 2*j+1, 2*k+1, 2*l))
                    term_set.add((2*i+1, 2*j, 2*k, 2*l))
                    term_set.add((2*i, 2*j+1, 2*k, 2*l))
                    term_set.add((2*i, 2*j, 2*k+1, 2*l))
                    term_set.add((2*i, 2*j, 2*k, 2*l+1))
    term_list = list(term_set)
    pool_list = [get_fermion_operator(MajoranaOperator(term)) for term in term_list]
    return pool_list
