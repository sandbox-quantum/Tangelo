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

from tangelo.linq import Circuit
from tangelo.linq.simulator import SimulatorBase
import tangelo.linq.translator as translator


class QDK(SimulatorBase):

    def __init__(self, n_shots=None, noise_model=None):
        super().__init__(target=1, n_shots=n_shots, noise_model=noise_model, backend_inform=self.backend_info)

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit: a circuit in the abstract format to be translated
                for the target backend.
            return_statevector(bool): option to return the statevector as well,
                if available.
            initial_statevector(list/array) : A valid statevector in the format
                supported by the target backend.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        translated_circuit = translator.translate_qsharp(source_circuit)
        with open('tmp_circuit.qs', 'w+') as f_out:
            f_out.write(translated_circuit)

        # Compile, import and call Q# operation to compute frequencies. Only import qsharp module if qdk is running
        # TODO: A try block to catch an exception at compile time, for Q#? Probably as an ImportError.
        import qsharp
        qsharp.reload()
        from MyNamespace import EstimateFrequencies
        frequencies_list = EstimateFrequencies.simulate(nQubits=source_circuit.width, nShots=self.n_shots)
        print("Q# frequency estimation with {0} shots: \n {1}".format(self.n_shots, frequencies_list))

        # Convert Q# output to frequency dictionary, apply threshold
        frequencies = {bin(i).split('b')[-1]: frequencies_list[i] for i, freq in enumerate(frequencies_list)}
        frequencies = {("0"*(source_circuit.width-len(k))+k)[::-1]: v for k, v in frequencies.items()
                       if v > self.freq_threshold}
        return (frequencies, None)

    @property
    def backend_info(self):
        return {"statevector_available": False, "statevector_order": None, "noisy_simulation": False}
