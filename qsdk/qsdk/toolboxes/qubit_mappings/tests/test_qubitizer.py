import unittest

from qsdk.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from qsdk.toolboxes.qubit_mappings import jordan_wigner

H2 = [("H", (0., 0., 0.)), ("H", (0., 0., 0.7414))]
molecule = SecondQuantizedMolecule(H2, 0, 0, "sto-3g")


def assert_term_dict_almost_equal(d1, d2, delta=1e-10):
    """ Utility function to check whether two qubit operators are almost equal, looking at their term dictionary,
    for an arbitrary absolute tolerance """
    d1k, d2k = set(d1.keys()), set(d2.keys())
    if d1k != d2k:
        d1_minus_d2 = d1k.difference(d2k)
        d2_minus_d1 = d2k.difference(d1k)
        raise AssertionError("Term dictionary keys differ. Qubit operators are not almost equal.\n"
                             f"d1-d2 keys: {d1_minus_d2} \nd2-d1 keys: {d2_minus_d1}")
    else:
        for k in d1k:
            if abs(d1[k] - d2[k]) > delta:
                raise AssertionError(f"Term {k}, difference={abs(d1[k]-d2[k])} > delta={delta}:\n {d1[k]} != {d2[k]}")


class QubitizerTest(unittest.TestCase):

    def test_qubit_hamiltonian_JW_h2(self):
        """ Verify computation of the Jordan-Wigner Hamiltonian for the H2 molecule """

        qubit_hamiltonian = jordan_wigner(molecule.fermionic_hamiltonian)

        # Obtained with Openfermion
        reference_terms = {(): -0.0988639693354571, ((0, 'Z'),): 0.17119774903432955, ((1, 'Z'),): 0.17119774903432958,
                           ((2, 'Z'),): -0.22278593040418496, ((3, 'Z'),): -0.22278593040418496,
                           ((0, 'Z'), (1, 'Z')): 0.16862219158920938, ((0, 'Z'), (2, 'Z')): 0.120544822053018,
                           ((0, 'Z'), (3, 'Z')): 0.165867024105892, ((1, 'Z'), (2, 'Z')): 0.165867024105892,
                           ((1, 'Z'), (3, 'Z')): 0.120544822053018, ((2, 'Z'), (3, 'Z')): 0.17434844185575687,
                           ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')): -0.045322202052874,
                           ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')): 0.045322202052874,
                           ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')): 0.045322202052874,
                           ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')): -0.045322202052874}

        assert_term_dict_almost_equal(qubit_hamiltonian.terms, reference_terms, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
