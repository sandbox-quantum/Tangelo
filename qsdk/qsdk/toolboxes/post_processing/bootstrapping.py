import numpy as np
from scipy import stats
from collections import Counter


def get_resampled_frequencies(freq_dict, ncount):
    """From a frequencies dictionary, makes a set of samples consistent with
    those frequencies and resample from that set to obtain a new frequencies
    dictionary consistent with ncount measurements.

    Args:
        freq_dict (array): dictionary of measurement/sample frequencies.
        ncount (int): number of shots/samples to generate resampled frequencies.

    Returns
        frequencies (dict): new frequencies dictionary with resampled
            distribution.
    """

    length_dict = len(freq_dict.keys())
    xk = np.empty(length_dict, dtype=np.int64)
    pk = np.empty(length_dict, dtype=float)

    # Convert to arrays of integers and frequencies and create generator for samples
    for i, (k, v) in enumerate(freq_dict.items()):
        xk[i] = int(k, 2)
        pk[i] = v
    distr = stats.rv_discrete(name="distr", values=(xk, pk))

    # Obtain output bitstring string format
    n_qubits = len(list(freq_dict.keys())[0])
    format_specifier = "0"+str(n_qubits)+"b"

    # Generate samples from distribution. Cut in chunks to ensure samples fit in memory, gradually accumulate
    chunk_size = 10**7
    n_chunks = ncount // chunk_size
    freqs_shots = Counter()

    for i in range(n_chunks+1):
        this_chunk = ncount % chunk_size if i == n_chunks else chunk_size
        samples = distr.rvs(size=this_chunk)
        freqs_shots += Counter(samples)
    frequencies = {format(k, format_specifier): v / ncount for k, v in freqs_shots.items()}

    return frequencies
