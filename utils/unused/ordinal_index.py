import numpy as np

def ordinal_index(octpy_metadata):
    num_trials = octpy_metadata.shape[0]
    trial_index_ordinal = np.arange(num_trials)
    
    return trial_index_ordinal