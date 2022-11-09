"""
    Simple example showing how this package can be used to implement variational algorithms such as VQE, in order
    to estimate the ground state energy of a molecule such as H2 in sto-3g basis.

    The following example assumes that a suitable state-preparation circuit and qubit operator have been provided
    by the user, obtained for example with a quantum chemistry package.
"""

import numpy as np
from scipy.optimize import minimize

from tangelo.linq import Gate, Circuit, get_backend


class H2StatePreparationCircuit(Circuit):

    def __init__(self):
        """
            Build the quantum circuit according to the user implementation of private function __build_circuit.
            The signature of this constructor is up to the user.
        """

        # Create an empty abstract circuit object, build it according to user code (here, in __build_circuit)
        Circuit.__init__(self, gates=None)
        self.__build_circuit()

    def __build_circuit(self):
        """ Hardcoded variational circuit derived through quantum chemistry packages, for H2 in sto-3g basis """
        half_pi = np.pi/2.
        
        mygates = list()
        mygates.append(Gate("X", 0))
        mygates.append(Gate("X", 1))
        mygates.append(Gate("RY", 0, parameter=half_pi))
        mygates.append(Gate("RX", 1, parameter=-half_pi))
        mygates.append(Gate("CNOT", 1, 0))
        mygates.append(Gate("RZ", 1, parameter="theta", is_variational=True))
        mygates.append(Gate("CNOT", 1, 0))
        mygates.append(Gate("RY", 0, parameter=-half_pi))
        mygates.append(Gate("RX", 1, parameter=half_pi))

        for g in mygates:
            self.add_gate(g)

    def set_var_params(self, var_params):
        """
            Take variational parameters as input and map their values to the parameters of variational gates.
            For seamless use with scipy, a list is a convenient input type.
        """
        # This variational circuit expects exactly one parameter
        assert(len(var_params) == 1)
        # The mapping is straightforward: there is only one variational gate and one parameter
        self._variational_gates[0].parameter = var_params[0]


# Instantiate the ansatz circuit
my_ansatz_circuit = H2StatePreparationCircuit()
print(my_ansatz_circuit)

# Hardcoded qubit Hamiltonian corresponding to H2 at equilibrium, in sto-3g basis, with qubit reduction
# Derived from quantum chemistry packages
from openfermion.ops import QubitOperator
H2_qb_hamiltonian = QubitOperator()
H2_qb_hamiltonian.terms = {(): -0.33992710211913446, ((0, 'Z'),): 0.39399763457689063, ((1, 'Z'),): 0.39399763457689063,
                           ((0, 'Z'), (1, 'Z')): 0.011236740873436901, ((0, 'X'), (1, 'X')): 0.18128753567242528}


def get_energy(var_params, simulation_backend, qb_ham, prep_circuit):
    """ Computes the expectation value of the qubit Hamiltonian, with regards to the parametric state preparation """
    prep_circuit.set_var_params(var_params)
    return simulation_backend.get_expectation_value(qb_ham, prep_circuit)


# Instantiate a simulator object for a target backend (try "qiskit" or "qulacs", for example)
my_simulator = get_backend(target="cirq")

# VQE attempts to reach ground state energy by minimizing the expectation value of the qubit Hamiltonian with regards
# to the parametric state preparation provided by the user. The initial parameters can be changed.
initial_var_params = [0.]
opt_result = minimize(get_energy, initial_var_params, args=(my_simulator, H2_qb_hamiltonian, my_ansatz_circuit),
                      method="COBYLA",
                      tol=0.0001,
                      options={'disp': True, 'maxiter': 5000, 'rhobeg': 0.05}
                      )

print(f"Energy obtained by VQE : {opt_result.fun} Ha, for var_params = {opt_result.x}")
