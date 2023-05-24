import numpy as np
import scipy
import warnings

def interpolate(input):
    """ replace np.NaN values with interpolated values for an input series """

    # allow list
    if type(input) == list:
        input = np.array(input)

    # require numpy array
    if not type(input) == np.ndarray:
        warnings.warn("Input must be list or numpy array")

        return None
    
    # return the array if data is already complete
    if np.isnan(input).sum() == 0:

        return input
    
    # remove nans
    nonNaNidx = np.flatnonzero(~np.isnan(input))
    f = scipy.interpolate.interp1d(nonNaNidx, input[nonNaNidx], \
                                    kind='linear', fill_value=np.nan, bounds_error=False)
    
    NaNidx = np.flatnonzero(np.isnan(input))
    input[NaNidx] = f(NaNidx)

        # fill leading or trailing NaNs with nearest non-NaN value
    mask = np.isnan(input)
    input[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), input[~mask])

    return input