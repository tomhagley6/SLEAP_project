import pandas as pd
import numpy as np

def correct_video_frame_reset(metadata):

    # create new Series from metadata with ordinal index
    corrected = pd.Series(metadata._frame.copy().values, index=np.arange(metadata._frame.values.shape[0]))
    # find ordinal index where _frame == 0
    frame_reset = corrected[corrected == 0].index.values[-1]
    
    # if index of frame == 0 is not 0 (beginning of video)
    if frame_reset != 0:
        # correct frames
        corrected[frame_reset:] = corrected[frame_reset:] + frame_reset
        # update original Dataframe
        metadata._frame = corrected.values


    return metadata