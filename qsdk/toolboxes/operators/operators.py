""" This module defines various kinds of operators used in vqe.
    It can later be broken down in several modules if needed. """

# Later on, if needed, we can extract the code for the operators themselves to remove the dependencies and customize
import openfermion


class FermionOperator(openfermion.FermionOperator):
    """ Currently, this class is coming from openfermion. Can be later on be replaced by our own implementation. """
    pass


class QubitOperator(openfermion.QubitOperator):
    """ Currently, this class is coming from openfermion. Can be later on be replaced by our own implementation. """

    def count_qubits(self):
        """ Return the number of qubits used by the qubit operator based on the highest index found in the terms."""
        if (len(self.terms.keys()) == 0) or ((len(self.terms.keys()) == 1) and (len(list(self.terms.keys())[0]) == 0)):
            return 0
        else:
            return max([(sorted(pw))[-1][0] for pw in self.terms.keys() if len(pw) > 0]) + 1
