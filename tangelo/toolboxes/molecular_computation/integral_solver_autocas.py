# Copyright 2023 Good Chemistry Company.
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

import os

import h5py
import numpy as np

from tangelo.toolboxes.molecular_computation.integral_solver import IntegralSolver


class IntegralSolverAutoCAS(IntegralSolver):
    """Electronic Structure integration for AutoCAS.

    TODO:
    Installation instruction and references.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
    This behaviour is the source of the following dependency conflicts.
    scine-autocas 2.1.0 requires numpy==1.19.5, but you have numpy 1.23.5 which is incompatible
    """

    def __init__(self, **settings):
        """ TODO
        """

        from scine_autocas import Autocas
        from scine_autocas.autocas_utils.molecule import Molecule
        from scine_autocas.interfaces.molcas import Molcas
        from scine_autocas.main_functions import MainFunctions

        self.autocas = Autocas
        self.molecule = Molecule
        self.molcas = Molcas
        self.functions = MainFunctions

        # Default settings.
        self.settings = {
            "root_folder": os.getcwd(),
            "project_name": "mol",
            "method": "dmrg_ci",
            "dmrg_bond_dimension": 250,
            "dmrg_sweeps": 5,
            "cholesky": True
        }
        self.settings.update(settings)

        self.xyz_file = os.path.join(self.settings["root_folder"], f"{self.settings['project_name']}.xyz")

    def set_physical_data(self, mol):
        """ TODO
        """

        # TODO check if file already exists.
        mol.to_file(self.xyz_file)

        self.autocas_mol = self.molecule(
            xyz_file=self.xyz_file,
            charge=mol.q,
            spin_multiplicity=mol.spin + 1, # TODO verify this.
            double_d_shell=True # TODO also verify this.
        )
        mol.n_electrons = self.autocas_mol.electrons

    def compute_mean_field(self, sqmol):
        """ TODO: add frozen_orbitals to the inplace change of the attributes.
        """

        autocas = self.autocas(self.autocas_mol)
        molcas = self.molcas([self.autocas_mol])

        # Setup interface
        molcas.project_name = self.settings["project_name"]
        molcas.settings.work_dir = os.path.join(self.settings["root_folder"], "molcas_work_dir")
        molcas.environment.molcas_scratch_dir = os.path.join(self.settings["root_folder"], "molcas_scratch_dir")
        molcas.settings.xyz_file = self.xyz_file
        molcas.settings.cholesky = self.settings["cholesky"]

        # Additional options related to the molecular problem.
        molcas.settings.basis_set = sqmol.basis
        molcas.settings.uhf = sqmol.uhf

        # TODO
        if sqmol.uhf == True:
            raise NotImplementedError

        # Default value for first pass (TODO: not hardcoded).
        molcas.settings.method = "dmrg_ci"
        molcas.settings.dmrg_bond_dimension = 250
        molcas.settings.dmrg_sweeps = 5

        # Make initial active space and evaluate initial DMRG calculation.
        occ_initial, index_initial = autocas.make_initial_active_space()

        # Initial HF calculation.
        molcas.calculate()

        # Returns energy, s1, s2 and mutual_information
        # https://github.com/qcscine/autocas/blob/67bf02866efdccb74abd40e06b56a7e8f4248ec7/scine_autocas/interfaces/molcas/__init__.py#L506-L531
        _, s1_entropy, _, _ = molcas.calculate(occ_initial, index_initial)

        # Make active space based on single orbital entropies.
        cas_occ, cas_index = autocas.get_active_space(
            occ_initial, s1_entropy,
            force_cas=True # Debug argument.
        )

        # Second DMRG options (with an active based on s1 entropies).
        molcas.settings.method = self.settings["method"]
        molcas.settings.dmrg_bond_dimension = self.settings["dmrg_bond_dimension"]
        molcas.settings.dmrg_sweeps = self.settings["dmrg_sweeps"]

        # Do a calculation with this CAS.
        final_energy, _, _, _ = molcas.calculate(cas_occ, cas_index)

        # Set SecondQuantizedMolecule attributes.
        sqmol.mf_energy = final_energy
        sqmol.mo_energies = molcas.hdf5_utils.mo_energies
        sqmol.mo_occ = molcas.hdf5_utils.occupations
        sqmol.n_mos = len(sqmol.mo_energies)
        sqmol.n_sos = sqmol.n_mos * 2

        with h5py.File(molcas.orbital_file, "r") as h5_file:
            mo_coeff = np.array(h5_file.get("MO_VECTORS"))
        self.mo_coeff = mo_coeff.reshape((sqmol.n_mos,)*2)

        print(sqmol.__dict__)
        sqmol.frozen_orbitals = [i for i in range(sqmol.n_mos) if i not in cas_index]

    def get_integrals(self, sqmol, mo_coeff=None):
        """ TODO
        """

        # TODO get integrals here.
        pass
