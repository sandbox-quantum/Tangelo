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

    def __init__(self, autocas_yaml=None, **settings):
        """ TODO
        """

        from scine_autocas.autocas_utils.molecule import Molecule
        from scine_autocas.main_functions import MainFunctions
        self.molecule = Molecule
        self.autocas_workflow = MainFunctions()

        self.root_folder = os.getcwd()

        self.autocas_yaml = autocas_yaml
        if autocas_yaml is None:
            self.autocas_yaml = os.path.join(self.root_folder, "autocas_settings.yml")

        # Default settings. Other settings can be modified, see this url for an
        # example: https://github.com/qcscine/autocas/blob/master/scripts/full.yml.
        self.settings = {
            "molecule": {
                "double_d_shell": True
            },
            "interface": {
                "interface": "molcas", # Theo nly one supported as of now.
                "project_name": "mol",
                "environment": {"molcas_scratch_dir": os.path.join(self.root_folder, "molcas_scratch")},
                "settings": {"work_dir":  os.path.join(self.root_folder, "autocas_project")}
            }
        }
        self.settings.update(settings)

        self.xyz_file = os.path.join(self.root_folder, f"{self.settings['interface']['project_name']}.xyz")
        self.settings["interface"]["settings"]["xyz_file"] = self.xyz_file
        self.settings["molecule"]["xyz_file"] = self.xyz_file


    def set_physical_data(self, mol):
        """ TODO
        """

        # TODO check if file already exists.
        mol.to_file(self.xyz_file)

        # Temporary molecule to file up the number of electrons.
        autocas_mol = self.molecule(
            xyz_file=self.xyz_file,
            charge=mol.q,
            spin_multiplicity=mol.spin + 1, # TODO verify this.
        )
        mol.n_electrons = autocas_mol.electrons

        self.settings["molecule"].update({
            "charge": mol.q,
            "spin_multiplicity": mol.spin + 1,
        })

    def compute_mean_field(self, sqmol):
        """ TODO: add frozen_orbitals to the inplace change of the attributes.
        """
        # PyYAML package is a requirement for autocas.
        import yaml

        #self.settings["interface"]["settings"]["basis_set"] = sqmol.basis

        # Additional options related to the molecular problem.
        # TODO
        if sqmol.uhf == True:
            raise NotImplementedError

        with open(self.autocas_yaml, "w", encoding="utf-8") as file:
            yaml.dump(self.settings, file)

        self.autocas_workflow.main({
            "yaml_input": self.autocas_yaml,
            "xyz_file": self.xyz_file,
            "basis_set": sqmol.basis
        })

        # Set SecondQuantizedMolecule attributes.
        sqmol.mf_energy = self.autocas_workflow.results["energy"]

        interface = self.autocas_workflow.results["interface"]
        sqmol.mo_energies = interface.hdf5_utils.mo_energies
        sqmol.mo_occ = interface.hdf5_utils.occupations

        sqmol.n_mos = len(sqmol.mo_energies)
        sqmol.n_sos = sqmol.n_mos * 2

        with h5py.File(interface.orbital_file, "r") as h5_file:
            mo_coeff = np.array(h5_file.get("MO_VECTORS"))
        self.mo_coeff = mo_coeff.reshape((sqmol.n_mos,)*2)

        cas_index = self.autocas_workflow.results["final_orbital_indices"]
        sqmol.frozen_orbitals = [i for i in range(sqmol.n_mos) if i not in cas_index]

    def get_integrals(self, sqmol, mo_coeff=None):
        """ TODO
        """
        # TODO get integrals here.
        # Here are some notes that I have gathered before this on the ice.
        # The one-body integrals are located in mol.OneInt. However, it contains
        # more than just the overlap integrals.
        # The two-body integrals are decomposed with the Cholesky method by
        # default. This is done by the SEWARD program.
        pass
