"""
    This example shows how users can perform surgery in their ProjectQ scripts to retrieve the circuit
    using the CommandPrinter engine, and then import it into qsdk.backendbuddy, to simulate it with something else
    such as a noisy simulator or a more performant / scalable simulator, such as qulacs.

    It relies on a function in the translator module called projectq2abs.
"""

from projectq import MainEngine
from projectq.ops import H, X, CNOT

# 1: Use CommandPrinter as a your ProjectQ engine backend. This engine just prints the commands run by ProjectQ.
# For more info about ProjectQ and how to swap and combine backends:
# https://github.com/1QB-Information-Technologies/Feynman/blob/master/1qbit_tutorial/1QBit_projectq_tutorial_python3.ipynb
from projectq.backends import CommandPrinter
eng = MainEngine(backend=CommandPrinter(accept_input=False))
# NB: if you're using openfermion-projectq, you might be using their uccsd trotter engine and have something like this:
# eng = uccsd_trotter_engine(compiler_backend=CommandPrinter(accept_input=False))


# 2: Before "simulation", swap standard output with StringIo object.
# The CommandPrinter engine will "print" to this object instead of your screen
import sys
from io import StringIO
s = StringIO()
sys.stdout, s = s, sys.stdout


# 3: Allocate, simulate, deallocate as usual.
# Here, I use a dummy circuit to construct a Bell pair, for the sake of example
# Only a subset of ProjectQ instructions are supported by qsdk.backendbuddy.
# NB: `Measure` instructions are currently ignored. This is only an issue if you have mid-circuit measurements.
qreg = eng.allocate_qureg(2)
eng.flush()
H | qreg[0]
CNOT | (qreg[1], qreg[0])
eng.flush(deallocate_qubits=True)


# 4: Restore standard output (swap back). We can grab the ProjectQ instructions from the StringIo object with `getvalue`
sys.stdout, s = s, sys.stdout
pq_circ = s.getvalue()
print(f"*** Your ProjectQ instructions, as shown by the CommandPrinter engine:\n{pq_circ}\n")


# 5: qsdk.backendbuddy can translate the projectq instructions into its abstract format
# The resulting circuit can then be translated in various formats or run on different compute backends.
from qsdk.backendbuddy import translator
abs_circ = translator._translate_projectq2abs(pq_circ)
print(f"*** The corresponding qsdk.backendbuddy abstract circuit:\n{abs_circ}\n")


# 6: Do your thing. Example: using qulacs as a simulation backend, using qsdk.backendbuddy.
# Check out 'the_basics.ipynb` python jupyter notebook for more info
from qsdk.backendbuddy import Simulator
cirq_sim = Simulator(target='cirq')
cirq_freqs = cirq_sim.simulate(abs_circ)
print(f"*** Simulation results (frequencies):\n{cirq_freqs}")
# For an expectation value, assume you have an openfermion-like operator called qubit_op
# cirq_expval = qulacs_sim.get_expectation_value(qubit_op, abs_circ)
# print(f"{cirq_expval}")
