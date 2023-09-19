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

"""Our own n-layered Integrated molecular Orbital and Molecular mechanics
(ONIOM) solver. User specifies either the number (if beginning from start of
list), or the indices, of atoms which are to be identified as the model
system(s), from the larger molecular system.

Main model class for running oniom-calculations. This is analogous to the
scf.RHF, etc. methods, requiring however a bit more information. User supplies
an atomic-geometry, and specifies the system, as well as necessary models,
of increasing sophistication.

Reference:
    - The ONIOM Method and Its Applications. Lung Wa Chung, W. M. C. Sameera,
    Romain Ramozzi, Alister J. Page, Miho Hatanaka, Galina P. Petrova,
    Travis V. Harris, Xin Li, Zhuofeng Ke, Fengyi Liu, Hai-Bei Li, Lina Ding
    and Keiji Morokuma
    Chemical Reviews 2015 115 (12), 5678-5796. DOI: 10.1021/cr5004419.
"""
from typing import List, Union, Tuple

from tangelo.helpers.utils import is_package_installed
from tangelo.problem_decomposition.problem_decomposition import ProblemDecomposition
from tangelo.problem_decomposition.oniom._helpers.helper_classes import Fragment
from tangelo.toolboxes.molecular_computation.molecule import atom_string_to_list, get_default_integral_solver


def get_default_mm_package():
    if is_package_installed("openmm"):
        return "openmm"
    elif is_package_installed("rdkit"):
        return "rdkit"
    else:
        return None


def get_charges_and_coords_from_pdb_file(filename: str, mmpackage: str = None):
    """Obtain the partial charges and geometry of a molecule given as a pdb file using mmpackage

    Args:
        filename (str): The location of the pdb file
        mmpackate (str): Either openmm or rdkit are supported.

    Returns:
        List[float]: The partial charges
        List[Tuple[str, Tuple[float, float, float]]]: The geometry of atom name and coordinates of the partial charges
    """

    try:
        import openbabel
        from openbabel.openbabel import obErrorLog
        obErrorLog.SetOutputLevel(0)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"openbabel is required to use get_charges_and_coords_from_pdb_file when supplying only a pdb file."
                                  "install with 'pip install openbabel-wheel'")

    # Obtain partial charges
    if mmpackage.lower() == "openmm":
        from openmm.app.pdbfile import PDBFile
        from openmm.app.forcefield import ForceField
        from openmm import NonbondedForce
        pdb = PDBFile(filename)
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology)
        nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
        charges = [nonbonded.getParticleParameters(i)[0]._value for i in range(system.getNumParticles())]
    elif mmpackage.lower() == "rdkit":
        import rdkit
        from rdkit.Chem import AllChem
        rdmol = rdkit.Chem.rdmolfiles.MolFromPDBFile(filename, removeHs=False)
        mmff_props = AllChem.MMFFGetMoleculeProperties(rdmol, mmffVariant="MMFF94")
        charges = [mmff_props.GetMMFFPartialCharge(i) for i in range(rdmol.GetNumAtoms())]

    # Obtain xyz string
    mol = openbabel.openbabel.OBMol()
    conv = openbabel.openbabel.OBConversion()
    conv.SetInAndOutFormats("pdb", "xyz")
    _ = conv.ReadFile(mol, filename)
    geometry = conv.WriteString(mol)
    geometry = geometry.split("\n", 2)[2:][0]
    geom = atom_string_to_list(geometry)

    return charges, geom


class QMMMProblemDecomposition(ProblemDecomposition):

    def __init__(self, opt_dict: dict):
        """Main class for the QMMM hybrid solver. It defines a QM region to be solved using quantum chemistry
        and an MM region solved via a force field.

        The QM region includes the partial charges from a force-field calculation.

        Attributes:
            geometry (string or list): XYZ atomic coords (in "str float float..."
                or [[str, (float, float, float)], ...] format) or a pdb file.
                If a pdb file is used, openMM must be installed.
            charges (List[(float, int)]): The charges used for the electrostatic embedding of the QM region.
                Each item in the list is a float corresponding to the charge.
                Not necessary if using a pdb file.
            coords (List[(float, float, float)]): The corresponding coordinates of the charges used for the electrostatic embedding.
                Each item in the list is a tuple of floats (x, y, z) for each partial charge from the MM region.
                Not necessary if using a pdb file.
            qmfragment (Fragment): Specification of the QM region its solvers.
            verbose (bolean): Verbose flag.
        """

        copt_dict = opt_dict.copy()
        self.geometry: Union[str, List[Tuple[str, Tuple[float]]]] = copt_dict.pop("geometry", None)
        self.charges: List(float) = copt_dict.pop("charges", None)
        self.coords: List[Tuple[float]] = copt_dict.pop("coords", None)
        if self.charges is not None and self.coords is not None and len(self.charges) != len(self.coords):
            raise ValueError(f"The length of the coords and charges lists must be the same but coords has length {len(self.coords)}"
                             f"while charges has length {len(self.charges)}")
        self.qmfragment: Fragment = copt_dict.pop("qmfragment", None)
        self.verbose: bool = copt_dict.pop("verbose", False)
        self.mmpackage: str = copt_dict.pop("mmpackage", get_default_mm_package())
        self.supported_mm_packages = ["openmm", "rdkit"]

        self.mmcharges: List[float] = []
        self.mmcoords: List[Tuple[float]] = []
        self.pdbgeometry = False

        if self.geometry[-3:] == "pdb":
            self.pdbgeometry = True
            if self.mmpackage is None:
                raise ModuleNotFoundError(f"Any of {self.supported_mm_packages} is required to use {self.__class__.__name__} when supplying only a pdb file")
            elif self.mmpackage.lower() not in self.supported_mm_packages:
                raise ValueError(f"{self.__class__.__name__} only supports the following MM packages {self.supported_mm_packages}")

            self.pdbcharges, self.geometry = get_charges_and_coords_from_pdb_file(self.geometry, self.mmpackage)
        if isinstance(self.charges, list) and self.coords is None:
            for filename in self.charges:
                if isinstance(filename, str) and filename[-3:] == "pdb":
                    charges, geometry = get_charges_and_coords_from_pdb_file(filename, self.mmpackage)
                    self.mmcharges += charges
                    self.mmcoords += [geom[1] for geom in geometry]

        if len(copt_dict.keys()) > 0:
            raise KeyError(f"Keywords :: {copt_dict.keys()}, not available in {self.__class__.__name__}.")

        # Raise error/warnings if input is not as expected
        if not self.geometry or not self.qmfragment:
            raise ValueError(f"A geometry and qm fragment must be provided when instantiating {self.__class__.__name__}.")

        self.geometry = atom_string_to_list(self.geometry) if isinstance(self.geometry, str) else self.geometry

        self.distribute_atoms()

        self.integral_solver = get_default_integral_solver(qmmm=True)(self.mmcoords, self.mmcharges)

        self.qmfragment.build(self.integral_solver)

    def distribute_atoms(self):
        """For each fragment, the atom selection is passed to the Fragment
        object. Depending on the input, the method has several behaviors.
        """

        if self.coords is not None:
            self.mmcoords += self.coords
            self.mmcharges += self.charges

        # Case when no atoms are selected -> whole system. i.e. no added partial charges
        if self.qmfragment.selected_atoms is None:
            self.qmfragment.geometry = self.geometry
        # Case where an int is detected -> first n atoms.
        elif type(self.qmfragment.selected_atoms) is int:
            self.qmfragment.geometry = self.geometry[:self.qmfragment.selected_atoms]
            if self.coords is None:
                self.mmcoords += [self.geometry[n][1] for n in range(self.qmfragment.selected_atoms)]
                self.mmcharges += self.pdbcharges[self.qmfragment.selected_atoms:]
        # Case where a list of int is detected -> atom indexes are selected.
        # First atom is 0.
        elif isinstance(self.qmfragment.selected_atoms, list) and all(isinstance(id_atom, int) for id_atom in self.qmfragment.selected_atoms):
            self.qmfragment.geometry = [self.geometry[n] for n in self.qmfragment.selected_atoms]
            if self.coords is None:
                self.mmcoords += [self.geometry[n][1] for n in range(len(self.geometry)) if n not in self.qmfragment.selected_atoms]
                self.mmcharges += [self.pdbcharges[n] for n in range(len(self.geometry)) if n not in self.qmfragment.selected_atoms]
        # Otherwise, an error is raised (list of float, str, etc.).
        else:
            raise TypeError("selected_atoms must be an int or a list of int.")

        # If there are broken_links (other than an empty list nor None).
        # The whole molecule geometry is needed to compute the position of
        # the capping atom (or functional group).
        if self.qmfragment.broken_links:
            for li in self.qmfragment.broken_links:
                self.qmfragment.geometry += li.relink(self.geometry)

    def simulate(self):
        r"""Run the QMMM electrostatic embedding calculation

        Returns:
            float: Total ONIOM energy.
        """

        return self.qmfragment.simulate()

    def get_resources(self):
        """Estimate the resources required by ONIOM. Only supports fragments
        solved with quantum solvers. Resources for each fragments are outputed
        as a dictionary.
        """

        quantum_resources = self.qmfragment.get_resources()

        if self.verbose:
            print(f"\t\t{quantum_resources}\n")

        return quantum_resources
