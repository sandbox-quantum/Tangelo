import numpy as np
from collections import Counter


def make_samples_box(freq_dict, ncount):
    """
    Make the box of samples for bootstrapping from the dictionary of frequencies
    and number of measurements
    We will draw from this box randomly for bootstrapping
    Args:
        freq_dict (dict): Dictionary of frequencies for each measurment
        ncount (int) : Number of measurements performed.
    Return:
        temp_box (size:ncount) a numpy array with values of keys in the dictionary
                          with number of them reflecting the histogram
    """

    current_start = 0
    # pre allocate samples box. All keys have the same length of string
    for key in freq_dict:
        datatype = '<U'+str(len(key))
    temp_box = np.empty(ncount, dtype=datatype)
    # Get probabilty of each elements from the dictionary and create proportional number in samples box
    for key, value in freq_dict.items():
        added_size = round(value*ncount)
        current_end = current_start + added_size
        if current_end != ncount:
            if abs(current_end - ncount) < 2:
                raise ValueError('Frequency dictionary sums to greater than 1')
            else:
                # most likely a finite precision error
                current_end = ncount
        temp_box[current_start:current_end] = key
        current_start += added_size

    return temp_box


def get_new_frequencies(freq_dict, ncount):
    """
    From a frequencies dictionary, makes a set of samples consistent with those frequencies and
    resample to obtain new frequencies dictionary consistent with ncount measurements.
    Args:
        freq_dict (array): dictionary of measurement/sample frequencies
        ncount (int): number of shots/samples to generate resampled frequencies

    Returns
        frequencies: new frequencies dictionary with resampled distribution
    """

    # regenerate measurements from frequencies
    original_sample = make_samples_box(freq_dict, ncount)

    # Get ncount random integers (between 0 and ncount-1)
    rand_indices = np.random.randint(ncount, size=ncount)

    # Pick out rand_indices from original to make resampled list of samples
    resample = [original_sample[newint] for newint in rand_indices]

    # generate new dictionary of samples
    frequencies = {k: v / ncount for k, v in Counter(resample).items()}

    return frequencies


def get_average_sd(energy_list):
    """
    Calculate the average and standard deviation from the energy distribution
    Args:
        energy_list (array): list of energy values
    Returns:
        Average and standard deviation of the energy distribution
    """

    energy_average = np.mean(energy_list)
    energy_standard_deviation = np.std(energy_list, ddof=1)
    return energy_average, energy_standard_deviation
