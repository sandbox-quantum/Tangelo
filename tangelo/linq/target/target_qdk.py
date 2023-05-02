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

import warnings

from tangelo.linq import Circuit
from tangelo.linq.target.backend import Backend
from tangelo.linq.translator import translate_circuit as translate_c


class QDKSimulator(Backend):

    def __init__(self, n_shots=None, noise_model=None):
        import qsharp
        super().__init__(n_shots=n_shots, noise_model=noise_model)
        self.qsharp = qsharp

    def simulate_circuit(self, source_circuit: Circuit, return_statevector=False, initial_statevector=None,
                         desired_meas_result=None, save_mid_circuit_meas=False):
        """Perform state preparation corresponding to the input circuit on the
        target backend, return the frequencies of the different observables, and
        either the statevector or None depending on the availability of the
        statevector and if return_statevector is set to True. For the
        statevector backends supporting it, an initial statevector can be
        provided to initialize the quantum state without simulating all the
        equivalent gates.

        Args:
            source_circuit (Circuit): a circuit in the abstract format to be translated
                for the target backend.
            return_statevector (bool): option to return the statevector as well,
                if available.
            initial_statevector (list/array) : A valid statevector in the format
                supported by the target backend.
            desired_meas_result (str): The binary string of the desired measurement.
                Must have the same length as the number of MEASURE gates in source_circuit
            save_mid_circuit_meas (bool): Save mid-circuit measurement results to
                self.mid_circuit_meas_freqs. All measurements will be saved to
                self.all_frequencies, with keys of length (n_meas + n_qubits).
                The leading n_meas values will hold the results of the MEASURE gates,
                ordered by their appearance in the source_circuit.
                The last n_qubits values will hold the measurements performed on
                each of qubits at the end of the circuit.

        Returns:
            dict: A dictionary mapping multi-qubit states to their corresponding
                frequency.
            numpy.array: The statevector, if available for the target backend
                and requested by the user (if not, set to None).
        """
        translated_circuit = translate_c(source_circuit, "qdk", output_options={"save_measurements": save_mid_circuit_meas})
        with open('tmp_circuit.qs', 'w+') as f_out:
            f_out.write(translated_circuit)

        if desired_meas_result:
            warnings.warn("qdk uses statistics from n_shots instead of statistics on the number of successful desired_meas_result.")

        n_meas = source_circuit.counts.get("MEASURE", 0)
        key_length = n_meas + source_circuit.width if save_mid_circuit_meas else source_circuit.width
        # Compile, import and call Q# operation to compute frequencies. Only import qsharp module if qdk is running
        # TODO: A try block to catch an exception at compile time, for Q#? Probably as an ImportError.
        self.qsharp.reload()
        from MyNamespace import EstimateFrequencies
        frequencies_list = EstimateFrequencies.simulate(nQubits=key_length, nShots=self.n_shots)
        print("Q# frequency estimation with {0} shots: \n {1}".format(self.n_shots, frequencies_list))

        # Convert Q# output to frequency dictionary, apply threshold
        frequencies = {bin(i).split('b')[-1]: freq for i, freq in enumerate(frequencies_list)}
        frequencies = {("0" * (key_length - len(k)) + k)[::-1]: v for k, v in frequencies.items()
                       if v > self.freq_threshold}
        self.all_frequencies = frequencies.copy()

        return (frequencies, None)

    @staticmethod
    def backend_info():
        return {"statevector_available": False, "statevector_order": None, "noisy_simulation": False}
