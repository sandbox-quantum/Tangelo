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

"""This python module stores template Q# strings that can be used to have Q#
code files ready to be shared with collaborators, submitted to a remote compute
backend through Microsoft services or compiled by the local QDK simulator.
"""

_header = '''namespace MyNamespace
{
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Extensions.Convert;
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Measurement;\n
'''

_qdk_template = '''
    /// # Summary:
    ///         Returns the estimated probabilities associated to the different measurements
    ///         after applying the state-preparation provided by the user at runtime to the quantum state
    operation EstimateFrequencies(nQubits : Int, nShots : Int) : Double[]
    {{
        mutable frequencies = new Double[2^nQubits];

        for (iShot in 1..nShots)
        {{
            // Apply Q# operation (state preparation to measurements) to qubit register of size nQubits
            mutable results = {operation_name}();

            // Update frequencies based on sample value
            mutable index = 0;
            for (iQubit in nQubits-1..-1..0)
            {{
                if (results[iQubit] == One) {{set index = index + 2^iQubit;}}
            }}

            set frequencies w/= index <- frequencies[index] + 1.0;
        }}

        // Rescale to obtain frequencies observed
        for (iFreq in 0..2^nQubits-1)
        {{
            set frequencies w/= iFreq <- frequencies[iFreq] / ToDouble(nShots);
        }}

        return frequencies;
    }}

    /// INSERT TRANSLATED CIRCUIT HERE
'''
