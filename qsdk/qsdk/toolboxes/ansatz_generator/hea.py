""" This module defines the hardware efficient ansatz class, for use in applying VQE """

import numpy as np

from .ansatz import Ansatz
from ._hea_circuit import HEACircuit
from qsdk.toolboxes.qubit_mappings.mapping_transform import get_qubit_number
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit


class HEA(Ansatz):
    """ This class implements the HEA ansatz.  """

    def __init__(self, molecule, mapping='jw', mean_field=None, up_then_down=False, n_layers=4, rottype='euler'):

        self.molecule = molecule
        self.mapping = mapping
        self.mf = mean_field
        self.up_then_down = up_then_down
        self.n_layers = n_layers
        self.rottype = rottype

        self.n_electrons = self.molecule.n_electrons
        self.n_spinorbitals = self.molecule.n_qubits  # This label makes no sense for some mappings but is correct
        self.n_qubits = get_qubit_number(mapping, self.n_spinorbitals)

        # Each euler layer has 3 variational parameters per qubit, and one non-variational entangler
        # Each real rottion layer has 1 variational parameter per qubit, and one non-variational entangler
        # There is an additional layer with no entangler.
        if self.rottype == 'euler':
            self.n_var_params = self.n_qubits * 3 * (self.n_layers + 1)
        elif self.rottype == 'real':
            self.n_var_params = self.n_qubits * 1 * (self.n_layers + 1)

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "zeros"}

        # Default initial parameters for initialization
        self.var_params_default = "random"
        self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, MP2 (...), providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if isinstance(var_params, str) and (var_params not in self.supported_initial_var_params):
            raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
        if var_params is None:
            var_params = self.var_params_default
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
            return get_reference_circuit(n_spinorbitals=self.molecule.n_qubits,
                                         n_electrons=self.molecule.n_electrons,
                                         mapping=self.mapping,
                                         up_then_down=self.up_then_down)

    def build_circuit(self, var_params=None):
        """Construct the variational circuit to be used as our ansatz."""
        self.var_params = self.set_var_params(var_params)

        reference_state_circuit = self.prepare_reference_state()

        hea_circuit = HEACircuit(self.n_qubits, self.n_layers, self.rottype)

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
