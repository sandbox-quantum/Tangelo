"""
    An example of how a user can define an ansatz circuit / variational form to use with a variational algorithm,
    for example. This class is a sub-class of the Circuit class: the user can access any of the method,
    attributes and properties of the Circuit class to help them implementing their ansatz circuit.

    The user is responsible for providing the recipe of how to build their variational circuit and for implementing
    the set_parameters method, which describes how an input list of variational parameters should be mapped
    to the parameters of the variational gates appearing in the circuit.

    If the variational form can be fully described with `n` parameters and is described as a circuit with `m`
    variational gates, a valid mapping takes in these `n` inputs, applies mathematical operations to them,
    computing each of the `m` values for the parameters appearing in the variational gates. By keeping `n` as small as
    possible, the user makes it easier and faster for variational algorithms to converge to an optimimal solution in
    general.

    Once the set_parameter method returns, the user should be able to translate or simulate their ansatz with
    a given backend immediately.

    The user is free to add their own extra methods to express more complex variational forms, requiring for
    example a circuit to set an initial state before applying the variational circuit, or requiring the ansatz to
    change significantly or "grow" (such as ADAPT-VQE).
"""

from tangelo.linq import Gate, Circuit


class MyAnsatzCircuit(Circuit):

    def __init__(self, n_singles, n_doubles):
        """
            Builds the quantum circuit according to the user implementation of private function __build_circuit.
            The signature of this constructor is up to the user.
        """

        # Create the underlying circuit object. The user is free to implement here an initial state preparation
        # If that makes sense to them, or can start with an empty circuit object.
        Circuit.__init__(self, gates=None)

        # The user can add their own attributes as they see fit, related to the nature of their ansatz circuit
        self.n_singles = n_singles
        self.n_doubles = n_doubles

        # Build the ansatz circuit at instantiation
        self.__build_circuit()

    def __build_circuit(self):
        """
            User specifies how the variational circuit must be built, using the Circuit method `add_gate`.
            Can be simple or elaborate, hardcoded or defined programmatically.
        """
        mygates = list()
        mygates.append(Gate("H", 2))
        mygates.append(Gate("RZ", 1, parameter="p0", is_variational=True))
        mygates.append(Gate("RZ", 1, parameter="p1", is_variational=True))
        mygates.append(Gate("CNOT", 1, 0))
        mygates.append(Gate("CNOT", 2, 1))
        mygates.append(Gate("Y", 0))
        mygates.append(Gate("RZ", 1, parameter="p2", is_variational=True))
        mygates.append(Gate("RZ", 1, parameter="p3", is_variational=True))

        for g in mygates:
            self.add_gate(g)

    def set_var_params(self, var_params):
        """
            The signature and input of this method should be compatible with optimizers such as the ones provided
            by scipy.

            User implements here how the list / array of input variational parameters maps to the parameters of the
            variational gates. The user could also implement this differently using symbolic expressions or other method
            if they desire to do so, they have complete control over this.

            This mapping can be simple or as elaborate as needed. The smaller the size of the input, the smaller the
            size of the parameter space is for the optimizer used in the variational algorithm.

            It is recommended that the user writes an assert statement verifying that the size of the input is
            as expected, before they attempt to map the parameters.
        """

        # Check if input is of the expected size. Specific to the ansatz itself.
        # Here we assume we have (n_singles + n_doubles) parameters
        assert(len(var_params) == self.n_singles + self.n_doubles)

        # Map the input variational parameters to the gate parameters
        # Below an example of non-trivial mapping that computes the values of gate parameters and maps them
        # to more than one gate sometimes, depending on their position in the input list
        for i in range(len(var_params)):
            if i < self.n_singles:
                # Assume here that the first n_singles parameters map directly to the n_singles first variational gates
                self._variational_gates[i].parameter = var_params[i]
            else:
                # Assume here that the remaining n_doubles input parameters are each mapped to 2 consecutive gates
                # and that the gate parameter is half the value of the corresponding input. Arbitrary example.
                index = self.n_singles + (i-self.n_singles)*2
                self._variational_gates[index].parameter = var_params[i] / 2
                self._variational_gates[index+1].parameter = var_params[i] / 2


# Instantiate the ansatz circuit
my_ansatz = MyAnsatzCircuit(n_singles=2, n_doubles=1)
print(my_ansatz)

# Set the parameters of the circuit (these parameter values can be coming from an optimizer for example)
my_ansatz.set_var_params([1., 2., 3.])
print(my_ansatz)

# Translate, simulate ...
