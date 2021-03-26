""" This module defines the reduced UCCs ansatz class (RUCC), providing the foundation to implement variational ansatz circuits """

import numpy as np

from agnostic_simulator import Circuit, Gate

from .ansatz import Ansatz


class RUCC(Ansatz):
    """ This class implements the reduced-UCC ansatz, i.e. UCC1=UCCD and UCC3=UCCSD. 
    Currently, only closed-shell is supported.
    This implies that the mean-field is computed with the RHF reference integrals. """

    def __init__(self, n_var_params=1):

        if (n_var_params == 1 or n_var_params == 3):
            pass
        else:
            raise ValueError("The number of parameters for RUCC must be 1 or 3.")
        self.n_var_params = n_var_params

        # Supported reference state initialization
        # TODO: support for others and compatibility with the qubit mapping used
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"zeros", "ones", "random"}

        # Default initial parameters for initialization
        self.var_params_initialization = "zeros"
        self.reference_state_initialization = "HF"

        self.var_params = None
        self.circuit = None

    def initialize_var_params(self):
        """ Compute sets of potential initial values for variational parameters, such as zeros, random numbers, MP2,
        or any insightful values. Impacts the convergence of the variational algorithm. """

        if self.var_params_initialization not in self.supported_initial_var_params:
            raise ValueError(f"Only supported initialization methods for variational parameters are:"
                             f"{self.supported_initial_var_params} ")

        if self.var_params_initialization == "zeros":
            initial_var_params = np.zeros((self.n_var_params,), dtype=np.float)
        if self.var_params_initialization == "ones":
            initial_var_params = np.ones((self.n_var_params,), dtype=np.float)
        elif self.var_params_initialization == "random":
            initial_var_params = np.random.random((self.n_var_params,), dtype=np.float)
        self.var_params = initial_var_params
        return initial_var_params

    # TODO: Possible initial states must be compatible with qubit transform used, add check for it later on
    def prepare_reference_state(self):
        """ Returns circuit preparing the reference state of the ansatz (e.g prepare reference wavefunction with HF,
        multi-reference state, etc). These preparations must be consistent with the transform used to obtain the
        qubit operator.
        """

        if self.reference_state_initialization not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        # NB: this one is consistent with JW but not other transforms.
        if self.reference_state_initialization == "HF":
            return Circuit([Gate("X", target=i) for i in [0,2]])
 
    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        # Set initial variational parameters used to build the circuit
        if var_params is not None: # Temporary, will be replaced once set_var_params can directly update the parameters
            assert(len(var_params) == self.n_var_params)
            self.var_params = var_params
        elif not self.var_params:
            self.initialize_var_params()

        # Prepare reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        if self.n_var_params == 1:
            rucc_circuit = self._ucc1()
        elif self.n_var_params == 3:
            rucc_circuit = self._ucc3()
        else:
            raise ValueError("The number of parameters for RUCC must be 1 or 3.")

        self.circuit = reference_state_circuit + rucc_circuit

    def update_var_params(self, var_params):
        """ Shortcut: set value of variational parameters in the already-built ansatz circuit member.
            The circuit does not need to be rebuilt every time if only the variational parameters change. """

        assert len(var_params) == self.n_var_params
        self.var_params = var_params

        for param_index in range(self.n_var_params):
            self.circuit._variational_gates[param_index].parameter = var_params[param_index]

    def _ucc1(self):
        """ This class implements the reduced-UCC ansatz UCC1.
        UCC1 is equivalent to the UCCD ansatz, but terms that act in the same manner of the HF state are removed. """

        # Initialization of an empty list.
        lst_gates = list()

        # UCC1 gates are appended.
        lst_gates.append(Gate("RX", 0, parameter=np.pi/2))

        for qubit_i in range(1,4):
            lst_gates.append(Gate("H", qubit_i))

        for qubit_i in range(3):
            lst_gates.append(Gate("CNOT", qubit_i+1, qubit_i))

        lst_gates.append(Gate("RZ", 3, parameter="theta", is_variational=True))

        for qubit_i in range(3,0,-1):
            lst_gates.append(Gate("CNOT", qubit_i, qubit_i-1))
            lst_gates.append(Gate("H", qubit_i))

        lst_gates.append(Gate("RX", 0, parameter=-np.pi/2))

        return Circuit(lst_gates)

    def _ucc3(self):
        """ This class implements the reduced-UCC ansatz UCC3.
        UCC3 is equivalent to the UCCSD ansatz, but terms that act in the same manner of the HF state are removed. """

        # Initialization of an empty list.
        lst_gates = list()

        # UCC3 gates are appended.
        lst_gates.append(Gate("RX", 0, parameter=np.pi/2))
        lst_gates.append(Gate("H", 1))
        lst_gates.append(Gate("RX", 2, parameter=np.pi/2))
        lst_gates.append(Gate("H", 3))

        lst_gates.append(Gate("CNOT", 1, 0))
        lst_gates.append(Gate("CNOT", 3, 2))

        lst_gates.append(Gate("RZ", 1, parameter="theta0", is_variational=True))
        lst_gates.append(Gate("RZ", 3, parameter="theta1", is_variational=True))

        lst_gates.append(Gate("CNOT", 3, 2))

        lst_gates.append(Gate("RX", 2, parameter=-np.pi/2))
        lst_gates.append(Gate("H", 2))

        lst_gates.append(Gate("CNOT", 2, 1))
        lst_gates.append(Gate("CNOT", 3, 2))

        lst_gates.append(Gate("RZ", 3, parameter="theta2", is_variational=True))

        for qubit_i in range(3,0,-1):
            lst_gates.append(Gate("CNOT", qubit_i, qubit_i-1))
            lst_gates.append(Gate("H", qubit_i))

        lst_gates.append(Gate("RX", 0, parameter=-np.pi/2))

        return Circuit(lst_gates)