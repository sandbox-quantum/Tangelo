namespace MyNamespace
{
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Extensions.Convert;
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Measurement;


    /// # Summary:
    ///         Returns the estimated probabilities associated to the different measurements
    ///         after applying the state-preparation provided by the user at runtime to the quantum state
    operation EstimateFrequencies(nQubits : Int, nShots : Int) : Double[]
    {
        mutable frequencies = new Double[2^nQubits];

        for (iShot in 1..nShots)
        {
            // Apply Q# operation (state preparation to measurements) to qubit register of size nQubits
            mutable results = MyQsharpOperation();

            // Update frequencies based on sample value
            mutable index = 0;
            for (iQubit in nQubits-1..-1..0)
            {
                if (results[iQubit] == One) {set index = index + 2^iQubit;}
            }

            set frequencies w/= index <- frequencies[index] + 1.0;
        }

        // Rescale to obtain frequencies observed
        for (iFreq in 0..2^nQubits-1)
        {
            set frequencies w/= iFreq <- frequencies[iFreq] / ToDouble(nShots);
        }

        return frequencies;
    }

    /// INSERT TRANSLATED CIRCUIT HERE
@EntryPoint()
operation MyQsharpOperation() : Result[] {
	mutable c = new Result[3];
	using (qreg = Qubit[3]) {
		H(qreg[2]);
		CNOT(qreg[0], qreg[1]);
		CNOT(qreg[1], qreg[2]);
		Y(qreg[0]);
		S(qreg[0]);
		Rx(2.0, qreg[1]);

		return ForEach(MResetZ, qreg);
	}
}

}
