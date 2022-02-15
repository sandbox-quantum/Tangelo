# Copyright 2021 Good Chemistry Company.
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

"""Helper functions on QubitOperators, such as exporting Openfermion
QubitOperator to file and vice-versa, or to visualize its coefficients for
example.
"""

import math
import numpy as np


def print_histogram_coeffs(qb_ham):
    """Convenience function printing a matplotlib histogram of the magnitudes
    of the coefficient in a QubitOperator object. Combine with the compress
    method of the QubitOperator class, this allows users to quickly identify
    what terms in the operator can be discarded, depending on the target
    accuracy of calculations.
    """

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    coefs = [max(abs(coef.real), abs(coef.imag)) for _, coef in qb_ham.terms.items()]
    magn = [math.floor(math.log10(coef)) for coef in coefs if coef != 0.]
    magn = np.array(magn)
    bins = list(np.arange(magn.min(), magn.max() + 2))

    plt.hist(magn, bins=bins, rwidth=.3)
    plt.xticks(ticks=bins, labels=[f"1e{i}" for i in bins])
    plt.grid(True)
    plt.xlabel('Magnitude')
    plt.title(f'Magnitude of qubit operator coeffs (Total = {len(coefs)})')
    plt.show()
