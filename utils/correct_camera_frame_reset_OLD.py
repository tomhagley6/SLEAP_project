import pandas as pd
import numpy as np

def correct_camera_frame_reset(start_frame_nums, stim_frame_nums, end_frame_nums):
    """ account for camera frame number resetting on the hour 
        find the first trial of camera reset, and correct for trial data
        for start, stim, and end frames of the trial
         
        Use this function when you are using a concatenated video for analysis """
    
    # solve bug by using this to find the index instead of get nearest timestamps? 
    test = pd.Series(end_frame_nums.values, index=np.arange(end_frame_nums.values.shape[0]))
    
    frame_types = [start_frame_nums, stim_frame_nums, end_frame_nums]
    
    # find timestamp where difference between consecutive frame numbers is negative
    # do this in the earliest set of timestamps, and apply to all sets
    frame_reset_ts = start_frame_nums[start_frame_nums.diff() < 0].index

    # if all trials are before camera frame reset, just return the frames unchanged
    if frame_reset_ts.size == 0:
        return start_frame_nums, stim_frame_nums, end_frame_nums


    for frame_type in frame_types:
        # find nearest timestamp to frame_reset_ts in this set of frames
        frame_type_frame_reset_ts = frame_type[frame_type.index.values >= frame_reset_ts.values[0]].index
        # get_loc to find index (must pass it a string or Invalidindexerror)
        frame_reset_loc = frame_type.index.get_loc(frame_type_frame_reset_ts.values[0])
        frame_num_at_reset = frame_type.iloc[frame_reset_loc -1]  # reset frame number
        reset_frames = frame_type.index.values >= frame_type_frame_reset_ts.values[0] # set of frames past reset
        frame_type[reset_frames] = frame_type[reset_frames] + frame_num_at_reset # correct to make frame_num cumulative


    return start_frame_nums, stim_frame_nums, end_frame_nums


