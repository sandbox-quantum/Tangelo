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

import abc
from typing import List, Tuple, Union, Type

from tangelo.helpers.utils import is_package_installed
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list


class MMChargesSolver(abc.ABC):
    """Instantiate electrostatic charges solver"""
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_charges(self) -> List[Tuple[float, Tuple[float, float, float]]]:
        """Obtain the charges for the given low parameters of the input fragment (e.g. filenames, list(s) of coordinates)

        Returns:
            List[float]: The list of charges
            List[Tuple(float, Tuple(float, float, float))]: The atoms and geometries of the charges"""
        pass


def convert_files_to_pdbs(filenames: Union[str, List[str]]):
    """Convert file or list of files to pdb files using openbabel.

    Args:
        filename (Union[str, List[str]]): The filename(s) that describe the MM region

    Returns:
        List[str]: The list of pdb equivalent filenames.
    """

    try:
        import openbabel
        from openbabel.openbabel import obErrorLog
        obErrorLog.SetOutputLevel(0)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"openbabel is required to use get_charges_and_coords_from_pdb_file when supplying only a pdb file."
                                  "install with 'pip install openbabel-wheel'")

    filesold = [filenames] if isinstance(filenames, str) else filenames

    mol = openbabel.openbabel.OBMol()
    conv = openbabel.openbabel.OBConversion()

    files = []
    for i, file in enumerate(filesold):
        if file[-3:] != "pdb":
            mol = openbabel.openbabel.OBMol()
            conv.SetOutFormat("pdb")
            conv.ReadFile(mol, file)
            conv.WriteFile(mol, f"tmp{i}.pdb")
            files += ["tmp{0}.pdb"]
        else:
            files += [file]

    return files


def get_default_mm_package() -> Type[MMChargesSolver]:
    if is_package_installed("rdkit"):
        return MMChargesSolverOpenMM()
    elif is_package_installed("openmm"):
        return MMChargesSolverRDKit()
    else:
        return None


def get_mm_package(package: str = "default"):
    """Return the MMCharges instance

    Args:
        package (str): The MM backend to use

    Returns:
        Type[MMChargesSolver]: The MMChargesSolver that computes the partial charges for given geometries
    """
    if package == "default":
        return get_default_mm_package()
    elif package.lower() == "openmm":
        return MMChargesSolverOpenMM()
    elif package.lower() == "rdkit":
        return MMChargesSolverRDKit()
    else:
        raise ValueError(f"Requested MM package of {package} is not currently supported in Tangelo")


class MMChargesSolverOpenMM(MMChargesSolver):
    def __init__(self, force_field_params=('amber14-all.xml', 'amber14/tip3pfb.xml')):
        self.force_field_params = force_field_params
        pass

    def get_charges(self, files) -> List[Tuple[float, Tuple[float, float, float]]]:
        from openmm.app.pdbfile import PDBFile
        from openmm.app.forcefield import ForceField
        from openmm import NonbondedForce
        from openmm.app import Modeller
        import openbabel

        pdbfiles = convert_files_to_pdbs(files)

        pdbs = [PDBFile(file) for file in pdbfiles]
        modeller = Modeller(pdbs[0].topology, pdbs[0].positions)
        for pdb in pdbs[1:]:
            modeller.add(pdb.topology, pdb.positions)
        forcefield = ForceField(*self.force_field_params)
        system = forcefield.createSystem(modeller.topology)
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        charges = [nonbonded.getParticleParameters(i)[0]._value for i in range(system.getNumParticles())]
        # Obtain xyz string
        mol = openbabel.openbabel.OBMol()
        conv = openbabel.openbabel.OBConversion()
        conv.SetInAndOutFormats("pdb", "xyz")
        geom = []
        for file in files:
            _ = conv.ReadFile(mol, file)
            geometry = conv.WriteString(mol)
            geom += atom_string_to_list(geometry.split("\n", 2)[2:][0])
        return charges, geom


class MMChargesSolverRDKit(MMChargesSolver):
    def __init__(self, mmffVariant="MMFF94"):
        self.mmffVariant = mmffVariant
        pass

    def get_charges(self, files) -> List[Tuple[float, Tuple[float, float, float]]]:
        import rdkit
        from rdkit.Chem import AllChem

        pdbfiles = convert_files_to_pdbs(files)
        rdmol = rdkit.Chem.rdmolfiles.MolFromPDBFile(pdbfiles[0], removeHs=False)
        for file in pdbfiles[1:]:
            mol_to_add = rdkit.Chem.rdmolfiles.MolFromPDBFile(file, removeHs=False)
            rdmol = rdkit.Chem.rdmolops.CombineMols(rdmol, mol_to_add)
        rdkit.Chem.SanitizeMol(rdmol)

        # Read charges and geometry
        mmff_props = AllChem.MMFFGetMoleculeProperties(rdmol, mmffVariant=self.mmffVariant)
        charges = [mmff_props.GetMMFFPartialCharge(i) for i in range(rdmol.GetNumAtoms())]
        rdkit.Chem.SanitizeMol(rdmol)
        new_xyz = rdkit.Chem.rdmolfiles.MolToXYZBlock(rdmol)
        # Strip first two lines and convert to standard format
        geom = atom_string_to_list("".join([val+"\n" for val in new_xyz.split('\n')[2:]]))

        return charges, geom
