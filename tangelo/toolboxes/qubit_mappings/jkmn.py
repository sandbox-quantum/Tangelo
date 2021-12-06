import numpy as np
from tangelo.toolboxes.operators import FermionOperator
from openfermion.transforms import get_majorana_operator
from ternary_tree import TST

create_ij = FermionOperator('0^ 1^') + FermionOperator('0 1')
mj = get_majorana_operator(create_ij)
print(mj)

tst = TST()


def numberToBase(n, b, h):
    if n == 0:
        return "0"*h
    digits = ""
    while n:
        digits += str(n % b)
        n //= b
    while len(digits) < h:
        digits += "0"
    return digits[::-1]


n = 4
h = int(np.log10(2*n+1)/np.log10(3))
for i in range(2*n):
    print(numberToBase(i, 3, 2))
    print(int(numberToBase(i, 3, 2), 3))

