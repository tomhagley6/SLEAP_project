"""

A script that produces a single function converting .bin files into pandas dataframes.
This function was created by NeuroGears

"""

# OS Libaries

import numpy as np
import pandas as pd

# A function to create pandas dataframes from .bin files written by NeuroG 

_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
                1 : np.dtype(np.uint8),
                2 : np.dtype(np.uint16),
                4 : np.dtype(np.uint32),
                8 : np.dtype(np.uint64),
                129 : np.dtype(np.int8),
                130 : np.dtype(np.int16),
                132 : np.dtype(np.int32),
                136 : np.dtype(np.int64),
                68 : np.dtype(np.float32)
                }

def read_harp_bin(file):
    """Reads data from the specified Harp binary file.
        file = read_harp_bin('/home/tomhagley/Documents/sleap_project/TaskLogic_10_2022-11-02T14-00-00.bin')
    print(file.head())aFrame
    
    """
    data = np.fromfile(file, dtype=np.uint8)
    
    if len(data) == 0:
        return None

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = _payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = ticks * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))

    if payload.shape[1] ==  1:
        ret_pd = pd.DataFrame(payload, index=seconds, columns= ["Value"])
        ret_pd.index.names = ['Seconds']

    else:
        ret_pd =  pd.DataFrame(payload, index=seconds)
        ret_pd.index.names = ['Seconds']

    return ret_pd

if __name__ == '__main__':
    file = read_harp_bin('/home/tomhagley/Documents/sleap_project/TaskLogic_10_2022-11-02T14-00-00.bin')
    print(file.head())