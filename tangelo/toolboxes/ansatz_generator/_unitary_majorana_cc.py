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

"""Utililty functions to generate pool of FermionOperators obtained from individual MajoranaOperators in
unitary coupled cluster expansions."""

from openfermion.transforms.opconversions.conversions import get_fermion_operator
from openfermion import MajoranaOperator
from numpy import integer

from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule


def get_majorana_uccsd_pool(molecule: SecondQuantizedMolecule = None, n_electrons: int = None, n_sos: int = None):
    """Construct a list of FermionOperator corresponding to the individual Majorana modes in the UCCSD ansatz

    Args:
        molecule (SecondQuantizedMolecule): The molecule to generate the Majorana pool from: Default None
        n_electrons (int): The number of active electrons: Default None
        n_sos (int): The number of active spin orbitals: Default None

    Returns:
        list: The list of FermionOperator for each Majorana operator in the UCCSD pool
    """

    if molecule is not None:
        n_active_electrons = molecule.n_active_electrons
        n_active_sos = molecule.n_active_sos
    elif isinstance(n_electrons, (int, integer)) and isinstance(n_sos, (int, integer)):
        n_active_electrons = n_electrons
        n_active_sos = n_sos
    else:
        raise ValueError("SecondQuantized mol or ints n_electrons/n_sos must be provided")

    def majorana_uccsd_generator():
        for i in range(n_active_electrons):
            for k in range(n_active_electrons, n_active_sos):
                yield (2*i, 2*k)
                yield (2*i+1, 2*k+1)
                for j in range(i+1, n_active_electrons):
                    for l in range(k+1, n_active_sos):
                        yield (2*i, 2*j+1, 2*k+1, 2*l+1)
                        yield (2*i+1, 2*j, 2*k+1, 2*l+1)
                        yield (2*i+1, 2*j+1, 2*k, 2*l+1)
                        yield (2*i+1, 2*j+1, 2*k+1, 2*l)
                        yield (2*i+1, 2*j, 2*k, 2*l)
                        yield (2*i, 2*j+1, 2*k, 2*l)
                        yield (2*i, 2*j, 2*k+1, 2*l)
                        yield (2*i, 2*j, 2*k, 2*l+1)

    pool_list = [get_fermion_operator(MajoranaOperator(term)) for term in majorana_uccsd_generator()]
    return pool_list


def get_majorana_uccgsd_pool(molecule: SecondQuantizedMolecule = None, n_sos: int = None):
    """Construct a list of FermionOperator corresponding to the individual Majorana modes in the UCCGSD ansatz

    Args:
        molecule (SecondQuantizedMolecule): The molecule to generate the Majorana pool from: Default None
        n_sos (int): The number of active spin orbitals: Default None

    Returns:
        list: The list of FermionOperator for each Majorana operator in the UCCGSD pool
    """

    if molecule is not None:
        n_active_sos = molecule.n_active_sos
    elif isinstance(n_sos, (int, integer)):
        n_active_sos = n_sos
    else:
        raise ValueError("SecondQuantized mol or int n_sos must be provided")

    def majorana_uccgsd_generator():
        for i in range(n_active_sos):
            for j in range(i+1, n_active_sos):
                yield (2*i, 2*j)
                yield (2*i+1, 2*j+1)
                for k in range(j+1, n_active_sos):
                    for l in range(k+1, n_active_sos):
                        yield (2*i, 2*j+1, 2*k+1, 2*l+1)
                        yield (2*i+1, 2*j, 2*k+1, 2*l+1)
                        yield (2*i+1, 2*j+1, 2*k, 2*l+1)
                        yield (2*i+1, 2*j+1, 2*k+1, 2*l)
                        yield (2*i+1, 2*j, 2*k, 2*l)
                        yield (2*i, 2*j+1, 2*k, 2*l)
                        yield (2*i, 2*j, 2*k+1, 2*l)
                        yield (2*i, 2*j, 2*k, 2*l+1)

    pool_list = [get_fermion_operator(MajoranaOperator(term)) for term in majorana_uccgsd_generator()]
    return pool_list
