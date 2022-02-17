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

# FNO, VS = 1, SCBK
#
# REFERENCE ENERGIES:
#     EQMF = -14.57233763 Eh
#     EILC = -14.60306939 Eh

import numpy as np

from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator
from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.qmf import QMF
from tangelo.toolboxes.ansatz_generator.ilc import ILC

sim = Simulator()

file_name = "./Be1_cc-pvdz_singlet.hdf5"

# Prepare classical data from SCF calculation
molecule = MolecularData(filename=file_name)
fermi_ham = get_fermion_operator(molecule.get_molecular_hamiltonian())
scfdata = (molecule, fermi_ham)

# Instantiate QMF ansatz -- note that SCBK mapping doesn't work for VS = 1 
qmf = QMF(molecule=None, mapping="SCBK", up_then_down=True, init_qmf=None, scfdata=scfdata)
qmf.set_var_params([np.pi, 0., 0., np.pi, 0., 0., 0., 0., 0., 0., 0., 0.])
qmf.build_circuit()
print(qmf.var_params)
print(qmf.circuit)

energy = sim.get_expectation_value(qmf.qubit_ham, qmf.circuit)
print(" EQMF = ", energy)
print(" EQMF (ref.) = -14.57233763")

ilc = ILC(molecule=None, mapping="SCBK", up_then_down=True, qmf_circuit=qmf.circuit, qmf_var_params=qmf.var_params,
          qubit_ham=qmf.qubit_ham, max_ilc_gens=None, n_trotter=1, scfdata=scfdata)
ilc.build_circuit()
#print(ilc.qmf_var_params)
#print(ilc.var_params)
#print(ilc.qmf_circuit)
#print(ilc.circuit)
energy = sim.get_expectation_value(ilc.qubit_ham, ilc.circuit)
print(" EILC (n_trot = 1) = ", energy)

ilc.n_trotter = 2
ilc.build_circuit()
# n_trotter = 2 is slightly better but not worth doubling the ilc circuit
energy = sim.get_expectation_value(ilc.qubit_ham, ilc.circuit)
print(" EILC (n_trot = 2) = ", energy)

ilc.n_trotter = 3
ilc.build_circuit()
# n_trotter = 3 is slightly better but not worth doubling the ilc circuit
energy = sim.get_expectation_value(ilc.qubit_ham, ilc.circuit)
print(" EILC (n_trot = 3) = ", energy)
print(" EILC (ref.) = -14.60306939")

