""" This module defines the hardware efficient ansatz class, for use in applying VQE as first defined in
    "Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets" 
    https://arxiv.org/abs/1704.05018 """

import numpy as np

from .ansatz import Ansatz
from ._hea_circuit import HEACircuit
from qsdk.toolboxes.qubit_mappings.mapping_transform import get_qubit_number
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from agnostic_simulator import Circuit


class HEA(Ansatz):
    """ This class implements the HEA ansatz.
        Args:
            molecule (MolecularData) : the molecular system
            mean-field (optional) : mean-field of molecular system
            up_then_down (bool): change basis ordering putting all spin up orbitals first, followed by all spin down
                                 Default, False has alternating spin up/down ordering.
            rot_type (str): 'euler' for RzRxRz on each qubit
                           'real' for Ry on each qubit
            n_layers (int): The number of HEA ansatz layers to use
                            One layer is hea_rot_type + grid of CNots
            reference_state (str): 'HF' for Hartree-Fock reference state,
                             'zero' for no reference state
        """

    def __init__(self, molecule=None, mapping='jw', mean_field=None, up_then_down=False, n_layers=2, rot_type='euler', n_qubits=None, reference_state=None):
        self.up_then_down = up_then_down
        self.n_layers = n_layers
        self.rot_type = rot_type

        if molecule:
            self.molecule = molecule
            self.mapping = mapping
            self.mf = mean_field
            self.n_electrons = self.molecule.n_electrons
            self.n_spinorbitals = self.molecule.n_qubits  # This label makes no sense for some mappings but is correct
            self.n_qubits = get_qubit_number(mapping, self.n_spinorbitals)

        if n_qubits:
            self.n_qubits = n_qubits

        # Each euler layer has 3 variational parameters per qubit, and one non-variational entangler
        # Each real rottion layer has 1 variational parameter per qubit, and one non-variational entangler
        # There is an additional layer with no entangler.
        if self.rot_type == 'euler':
            self.n_var_params = self.n_qubits * 3 * (self.n_layers + 1)
        elif self.rot_type == 'real':
            self.n_var_params = self.n_qubits * 1 * (self.n_layers + 1)
        else:
            raise ValueError("Supported rot_type is 'euler' and 'real'")

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF", "zero"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "zeros"}

        # Default initial parameters for initialization
        self.var_params_default = "random"
        if reference_state:
            self.default_reference_state = reference_state
        else:
            self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, MP2 (...), providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            else:
                if var_params == "ones":
                    initial_var_params = np.ones((self.n_var_params,), dtype=float)
                elif var_params == "random":
                    initial_var_params = 4 * np.pi * (np.random.random((self.n_var_params,)) - 0.5)
                elif var_params == "zeros":
                    initial_var_params = np.zeros((self.n_var_params,), dtype=float)
        else:
            try:
                assert (len(var_params) == self.n_var_params)
                initial_var_params = np.array(var_params)
            except AssertionError:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but received {len(var_params)}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """Prepare a circuit generating the HF reference state."""
        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.default_reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                         n_electrons=self.n_electrons,
                                         mapping=self.mapping,
                                         up_then_down=self.up_then_down)
        if self.default_reference_state == "zero":
            return Circuit()

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""
        self.var_params = self.set_var_params(var_params)

        reference_state_circuit = self.prepare_reference_state()

        hea_circuit = HEACircuit(self.n_qubits, self.n_layers, self.rot_type)

        if reference_state_circuit.size != 0:
            self.circuit = hea_circuit + reference_state_circuit
        else:
            self.circuit = hea_circuit

        self.update_var_params(self.var_params)
        return self.circuit

    def update_var_params(self, var_params):
        """Update variational parameters (done repeatedly during VQE)"""
        self.set_var_params(var_params)
        var_params = self.var_params

        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]
