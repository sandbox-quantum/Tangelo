# Copyright 2023 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Test class to keep track of performance of translator module, for format convertion """

import unittest
import os
import time

from itertools import product
import pytest

from tangelo.linq import Gate, Circuit
from tangelo.toolboxes.operators import QubitOperator
from tangelo.helpers.utils import symbolic_backends
from tangelo.linq import translate_operator, translate_circuit
from tangelo.linq.translator.translate_qubitop import FROM_TANGELO as FROM_TANGELO_OP
from tangelo.linq.translator.translate_qubitop import TO_TANGELO as TO_TANGELO_OP
from tangelo.linq.translator.translate_circuit import FROM_TANGELO as FROM_TANGELO_C
from tangelo.linq.translator.translate_circuit import TO_TANGELO as TO_TANGELO_C


# Build artificially large operator made of all possible "full" Pauli words (no 'I') of length n_qubits
n_qubits_op = 10
n_terms = 3 ** n_qubits_op
terms = {tuple(zip(range(n_qubits_op), pw)): 1.0 for pw in product(['X', 'Y', 'Z'], repeat=n_qubits_op)}
tangelo_op = QubitOperator()
tangelo_op.terms = terms

# Build artificially large quantum circuit
n_qubits = 20
n_repeat = 4000
gates = [Gate('X', i) for i in range(n_qubits)] + [Gate("CNOT", (i+1) % n_qubits, control=i) for i in range(n_qubits)]
tangelo_c = Circuit(gates * n_repeat)


class PerfTranslatorTest(unittest.TestCase):

    @pytest.mark.skip(reason="Takes a long time and doesn't print the desired information.")
    def test_perf_operator(self):
        """ Performance test with a reasonable large input for operator.
        Symbolic backends are not included in this test.
        """

        print(f'\n[Performance Test :: linq operator format conversion]')
        print(f'\tInput size: n_qubits={n_qubits_op}, n_terms={n_terms}\n')

        perf_backends = FROM_TANGELO_OP.keys() - symbolic_backends
        for f in perf_backends:
            try:
                tstart = time.time()
                target_op = translate_operator(tangelo_op, source="tangelo", target=f)
                print(f"\tFormat conversion from {'tangelo':12} to {f:12} :: {time.time()-tstart} s")
            except Exception:
                continue

            if f in TO_TANGELO_OP:
                try:
                    tstart = time.time()
                    translate_operator(target_op, source=f, target="tangelo")
                    print(f"\tFormat conversion from {f:12} to {'tangelo':12} :: {time.time()-tstart} s")
                except Exception:
                    continue

    @pytest.mark.skip(reason="Takes a long time and doesn't print the desired information.")
    def test_perf_circuit(self):
        """ Performance test with a reasonable large input for quantum circuit.
        Symbolic backends are not included in this test.
        """

        print(f'\n[Performance Test :: linq circuit format conversion]')
        print(f'\tInput size: n_qubits={tangelo_c.width}, n_gates={tangelo_c.size}\n')

        perf_backends = FROM_TANGELO_C.keys() - symbolic_backends
        for f in perf_backends:
            try:
                tstart = time.time()
                target_c = translate_circuit(tangelo_c, source="tangelo", target=f)
                print(f"\tFormat conversion from {'tangelo':12} to {f:12} :: {time.time()-tstart} s")
            except Exception:
                continue

            if f in TO_TANGELO_C:
                try:
                    tstart = time.time()
                    translate_circuit(target_c, source=f, target="tangelo")
                    print(f"\tFormat conversion from {f:12} to {'tangelo':12} :: {time.time()-tstart} s")
                except Exception:
                    continue


if __name__ == "__main__":
    unittest.main()
