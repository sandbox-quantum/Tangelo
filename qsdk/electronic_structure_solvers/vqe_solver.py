"""
Class providing the means to run the variational quantum eigensolver (VQE)
algorithm to solve electronic structure calculations.
"""

from enum import Enum
import numpy as np
from copy import deepcopy


class Ansatze(Enum):
    """ Enumeration of the ansatz circuits supported by VQE"""
    UCCSD = 0


# TODO: VQESolver to inherit from an abstract "ElectronicStructureSolver" class implementing common interface for PD
class VQESolver:
    """ Solve the electronic structure problem for a molecular system by using the
    variational quantum eigensolver (VQE) algorithm.
    It requires to choose an ansatz (variational form), a classical optimizer,
    initial variational parameters and a hardware backend to carry the quantum circuit
    simulation. Default values for these are provided by QEMIST, and can be overridden
    as depicted in tests and examples.
    In order to use this class, users must first set up the desired attributes
    of the VQESolver object and call the "build" method to build the underlying objects
    (hardware backend, ansatz...). They are then able to call any of the
    energy_estimation, simulate, or get_rdm methods. In particular, the simulate method
    runs the VQE algorithms, returning the optimal energy found by the classical optimizer.

    Attributes:
        backend_parameters (dict): A dictionary describing the target compute backend to use
            (ex: qulacs, qiskit, qdk, projectq...), the number of shots or noise model to use.
        backend (agnostic_simulator.Simulator): The Simulator object as implemented in the
            agnostic_simulator package, after "build" has been called.
        ansatz_type (Ansatze): Type of the desired ansatz, from the available ones.
        ansatz (vqe.Ansatz): An implementation of the abstract Ansatz class from vqe
        optimizer (function): Function performing classical optimization.
        initial_var_params (list): Initial values of the variational parameters
            used in the classical optimization process
        optimal_var_params (list): Parameters returning the optimal energy found during
            the optimization process.
        optimal_energy (float): Optimal energy found during the optimization process
        verbose (boolean): Controls the verbosity of the VQE methods.
    """

    def __init__(self):
        self.verbose = True

        self.ansatz_type = None
        self.ansatz = None
        self.qubit_mapping = "JW"

        self.optimizer = None
        self.initial_var_params = None

        self.backend = None
        self.backend_parameters = {}

        self.optimal_energy = None
        self.optimal_var_params = None

    def build(self, molecule, mean_field=None, frozen_orbitals=None):
        """ Build the underlying objects required to run ot build the ansatz
        circuit and run the VQE algorithm
        Args:
             molecule (pyscf.gro.mole): A pyscf molecule
             mean_field (): mean-field of the molecular system
             frozen_orbitals (list, tuple, int): The frozen orbitals.
                 If list: A complete list of frozen orbitals.
                 If int: The first n-number of occupied orbitals are frozen.
                 If tuple: The first n-number of occupied and last n-number of
                     virtual orbitals are frozen.
        """
        from agnostic_simulator import Simulator
        from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
        from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
        from qsdk.toolboxes.qubit_mappings import jordan_wigner
        from qsdk.toolboxes.ansatz_generator.uccsd import UCCSD

        # Ensure inputs have valid values before moving forward
        if not mean_field:
            mean_field = prepare_mf_RHF(molecule)
        if frozen_orbitals is not None:
            print("VQESolver does not support frozen orbitals at this time.")

        # Compute qubit hamiltonian for the input molecular system
        self.qemist_molecule = MolecularData(molecule)
        self.fermionic_hamiltonian = self.qemist_molecule.get_molecular_hamiltonian()
        self.qubit_hamiltonian = jordan_wigner(self.fermionic_hamiltonian)

        if self.verbose:
            n_qubits = self.qubit_hamiltonian.count_qubits()
            print(f"VQE qubit hamiltonian ::\tn_qubits = {n_qubits}\tn_terms = {len(self.qubit_hamiltonian.terms)}")

        # Build ansatz circuit according to input (mean field only used for MP2)
        if self.ansatz_type == Ansatze.UCCSD:
            self.ansatz = UCCSD(self.qemist_molecule, mean_field)

        # Build ansatz circuit with default initial parameters
        if not self.ansatz.circuit:
            self.ansatz.build_circuit()

        # If no classical optimizer was provided, assign a default one from scipy
        if not self.optimizer:
            self.optimizer = self._default_optimizer

        # Initial variational parameters
        if self.initial_var_params:
            try:
                assert (len(self.initial_var_params) == self.ansatz.n_var_params)
            except:
                raise ValueError(f"Expected {self.ansatz.n_var_params} variational parameter "
                                 f"but received {len(self.initial_var_params)}.")
        else:
            self.initial_var_params = self.ansatz.initialize_var_params()
        if self.verbose:
            print(f"VQE Number of variational parameters = {len(self.initial_var_params)}\n")

        # Quantum circuit simulation backend options
        target = self.backend_parameters.get("target", "qulacs")
        n_shots = self.backend_parameters.get("n_shots", None)
        noise_model = self.backend_parameters.get("noise_model", None)
        print(f"VQE Hardware backend : {target}, shots = {n_shots}")
        self.backend = Simulator(target=target, n_shots=n_shots, noise_model=noise_model)

    def simulate(self):
        """ Run the VQE algorithm, using the ansatz, classical optimizer, initial parameters and
         hardware backend built in the build method """
        if not (self.ansatz or self.backend):
            raise RuntimeError("No ansatz circuit or hardware backend built. Have you called VQESolver.build ?")
        return self.optimizer(self.energy_estimation, self.initial_var_params)

    def energy_estimation(self, var_params):
        """ Estimate energy using the given ansatz, qubit hamiltonian and compute backend.
         Keeps track of optimal energy and variational parameters along the way

        Args:
             var_params (numpy.array or list): variational parameters to use for VQE energy evaluation
        Returns:
             energy (float): energy computed by VQE using the ansatz and input variational parameters
        """

        # Update variational parameters, compute energy using the hardware backend
        var_params = np.array(var_params)
        self.ansatz.update_var_params(var_params)
        energy = self.backend.get_expectation_value(self.qubit_hamiltonian, self.ansatz.circuit)

        if self.verbose:
            print(f"\tEnergy = {energy:.7f} ")
        if (self.optimal_energy and (energy < self.optimal_energy)) or not self.optimal_energy:
            self.optimal_energy = energy
            self.optimal_var_params = var_params

        return energy

    def get_rdm(self, var_params):
        """ Compute the 1- and 2- RDM matrices using the VQE energy evaluation. This method allows
        to combine the DMET problem decomposition technique with the VQE as an electronic structure solver.
         The RDMs are computed by using each fermionic Hamiltonian term, transforming them and computing
         the elements one-by-one.
         Note that the Hamiltonian coefficients will not be multiplied as in the energy evaluation.
         The first element of the Hamiltonian is the nuclear repulsion energy term,
         not the Hamiltonian term.

         Args:
             var_params (numpy.array or list): variational parameters to use for VQE energy evaluation
         Returns:
             (numpy.array, numpy.array): One & two-particle RDMs (rdm1_np & rdm2_np, float64).
         """
        from qsdk.toolboxes.qubit_mappings import jordan_wigner

        # Save our accurate hamiltonian
        tmp_hamiltonian = self.qubit_hamiltonian

        # Initialize the RDM arrays
        n_mol_orbitals = len(self.ansatz.mf.mo_energy)
        rdm1_np = np.zeros((n_mol_orbitals,) * 2)
        rdm2_np = np.zeros((n_mol_orbitals,) * 4)

        # Lookup "dictionary" (lists are used because keys are non-hashable) to avoid redundant computation
        lookup_ham, lookup_val = list(), list()

        # Loop over each element of Hamiltonian (non-zero value)
        for ikey, key in enumerate(self.fermionic_hamiltonian):
            length = len(key)
            # Ignore constant / empty term
            if not key:
                continue
            # Assign indices depending on one- or two-body term
            if (length == 2):
                iele, jele = (int(ele[0]) // 2 for ele in tuple(key[0:2]))
            elif (length == 4):
                iele, jele, kele, lele = (int(ele[0]) // 2 for ele in tuple(key[0:4]))

            # Select the Hamiltonian element (Set coefficient to one)
            hamiltonian_temp = deepcopy(self.fermionic_hamiltonian)
            for key2 in hamiltonian_temp:
                hamiltonian_temp[key2] = 1. if (key == key2 and ikey != 0) else 0.

            # Obtain qubit Hamiltonian
            qubit_hamiltonian2 = jordan_wigner(hamiltonian_temp)
            qubit_hamiltonian2.compress()

            if qubit_hamiltonian2.terms in lookup_ham:
                opt_energy2 = lookup_val[lookup_ham.index(qubit_hamiltonian2.terms)]
            else:
                # Overwrite with the temp hamiltonian, use it to calculate the energy, store in lookup lists
                self.qubit_hamiltonian = qubit_hamiltonian2
                opt_energy2 = self.energy_estimation(var_params)
                lookup_ham.append(qubit_hamiltonian2.terms)
                lookup_val.append(opt_energy2)

            # Put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_np[iele, jele] += opt_energy2
            elif length == 4:
                if iele != lele or jele != kele:
                    rdm2_np[lele, iele, kele, jele] += 0.5 * opt_energy2
                    rdm2_np[iele, lele, jele, kele] += 0.5 * opt_energy2
                else:
                    rdm2_np[iele, lele, jele, kele] += opt_energy2

        # Restore the accurate hamiltonian
        self.qubit_hamiltonian = tmp_hamiltonian

        return rdm1_np, rdm2_np

    def serialize(self) -> dict:
        """Returns a serialized version of the solver.
        The serialized version of the solver should be all that's needed to
        reconstruct the solver.
        Returns:
            dict: Dictionary mapping the attributes of the solver to the values
                that they take.
        """
        import pickle
        return {
            "next_solver": "VQESolver",
            "solver_params": {
                "backend_parameters":  pickle.dumps(self.backend_parameters),
                "ansatz_type":                  pickle.dumps(self.ansatz_type),
                "optimizer":                    pickle.dumps(self.optimizer),
                "initial_var_params":           self.initial_var_params,
                "verbose":                      self.verbose,
            }
        }

    @staticmethod
    def load_serial(serialized_dict: dict) -> '__class__':
        """Builds an instance from a serialized representation.
        Args:
            serialized_dict(dict): The data required to builld the object.
        Returns:
            VQESolver: The solver instance.
        """
        import pickle

        solver = VQESolver()
        solver.backend_parameters = pickle.loads(serialized_dict["backend_parameters"])
        solver.ansatz_type = pickle.loads(serialized_dict["ansatz_type"])
        solver.optimizer = pickle.loads(serialized_dict["optimizer"])
        solver.initial_var_params = serialized_dict["initial_var_params"]
        solver.verbose = serialized_dict["verbose"]

        return solver

    def _default_optimizer(self, func, var_params):
        """ Function used as a default optimizer when user does not provide one.
        Args:
            func (function handle): A function to perform energy estimation.
             Takes var_params as input and returns a float.
            var_params (list): The variational parameters (float64).
        Returns:
            list: The new variational parameters (result.fun, float64).
        """

        from scipy.optimize import minimize
        result = minimize(func, var_params, method='SLSQP',
                          options={'disp': True, 'maxiter': 2000, 'eps': 1e-5, 'ftol': 1e-5})
        if self.verbose:
            print("\n\t\tOptimal UCCSD Singlet Energy: {}".format(result.fun))
            print("\t\tOptimal UCCSD Singlet Amplitudes: {}".format(result.x))
            print("\t\tNumber of Function Evaluations : ", result.nfev)

        return result.fun
