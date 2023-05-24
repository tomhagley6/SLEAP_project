import pandas as pd
import numpy as np
import find_frames

    # double check that stim_on time is accurate from bonsai hardware
def response_times_session(octpy_metadata):
    """ return response times for each trial in session """

    # params
    om = octpy_metadata

    # trial_start times
    trial_start_times = pd.Series(om.index.values)
    # stim_start times
    om['stim_start_time'] = om['ts'] + om['stimulus_delay'].apply(pd.to_timedelta, unit='S')
    stim_start_times = om.reset_index().stim_start_time # replace trial_start_time index with ordinal index

    # trial end
    response_times_naive = om.reset_index().RT
    response_times = response_times_naive.apply(pd.to_timedelta, unit='S') - (stim_start_times - trial_start_times)
    response_times = response_times/pd.Timedelta(1,'s')
    

    return response_times

# test
if __name__ == '__main__':
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'

    octpy_metadata, _ = find_frames.access_metadata(data_root, session)
    response_times = response_times_session(octpy_metadata)