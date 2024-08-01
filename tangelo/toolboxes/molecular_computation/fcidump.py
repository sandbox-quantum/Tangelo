# Copyright SandboxAQ 2021-2024.
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

from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from pyscf import ao2mo
from pyscf.tools import fcidump

from tangelo.toolboxes.qubit_mappings.mapping_transform import get_fermion_operator


def fermop_from_fcidump(filename):
    """Generate a Fermionic operator from an FCIDUMP file.

    This function reads an FCIDUMP file, which contains the one- and
    two-electron integrals of the electronic Hamiltonian, and constructs a
    Fermionic operator that represents the Hamiltonian of the system. Useful
    when only a FCIDUMP file is available, i.e. with no molecular coordinates,
    charge nor spin.

    Args:
        filename (str): The name of the FCIDUMP file to read. This file should
            contain the one-body and two-body integrals in the FCIDUMP format.

    Returns:
        tangelo.toolboxes.operators.FermionOperator: A Fermionic operator
            representing the Hamiltonian of the system, constructed from the
            data in the FCIDUMP file.
    """

    # Reading the FCIDUMP file.
    fcidump_data = fcidump.read(filename)

    # Getting the relevant data, like number of orbitals, core constant, and
    # integrals.
    norb = fcidump_data["NORB"]
    core_constant = fcidump_data["ECORE"]
    one_body_integrals = fcidump_data["H1"].reshape((norb,)*2)
    two_body_integrals = fcidump_data["H2"]
    two_body_integrals = ao2mo.restore(1, two_body_integrals, norb)
    two_body_integrals = two_body_integrals.transpose(0, 2, 3, 1)

    # Constructing the fermionic operator object.
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_body_integrals, two_body_integrals)
    molecular_operator = InteractionOperator(core_constant, one_body_coefficients, 0.5 * two_body_coefficients)

    return get_fermion_operator(molecular_operator)
