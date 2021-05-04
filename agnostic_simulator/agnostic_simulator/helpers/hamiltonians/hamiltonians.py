""" helper function: exporting Openfermion Hamiltonian to file and vice-versa """

from openfermion.ops import QubitOperator


def ham_of_to_string(of_qb_ham):
    """ Converts an Openfermion QubitOperator into a string with information for a Pauli word per line """
    res = ""
    for k, v in of_qb_ham.terms.items():
        res += f'{str(v)}\t{str(k)}\n'
    return res


def string_ham_to_of(string_ham):
    """ Reverse function of ham_of_to_string : reads a Hamiltonian from a file that uses the Openfermion syntax,
     loads it into an openfermion QubitOperator """
    of_terms_dict = dict()
    string_ham = string_ham.split('\n')[:-1]

    for term in string_ham:
        coef, word = term.split('\t')
        of_terms_dict[eval(word)] = float(coef)

    res = QubitOperator()
    res.terms = of_terms_dict
    return res
