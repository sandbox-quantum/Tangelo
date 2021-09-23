"""
    This python module stores template Q# strings that can be used to have Q# code files ready
    to be shared with collaborators, submitted to a remote compute backend through Microsoft services
    or compiled by the local QDK simulator.
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
